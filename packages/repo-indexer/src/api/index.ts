import { createHash } from 'node:crypto'
import {
  mkdir,
  readFile,
  readdir,
  rename,
  stat,
  writeFile,
} from 'node:fs/promises'
import path from 'node:path'
import { performance } from 'node:perf_hooks'

import { Queue } from 'bullmq'
import { Hono } from 'hono'
import IORedis from 'ioredis'
import { z } from 'zod'

import { defaultRetrievalBudget, retrieveIndexedChunks } from '../retrieval'
import type { IndexingRequest, RepoIndexStore, RetrievalBudget } from '../types'

export const IndexingRequestSchema = z.object({
  projectId: z.string().min(1),
  repoPath: z.string().min(1),
  workflowId: z.string().min(1).optional(),
  requestId: z.string().min(1).optional(),
  includeGlobs: z.array(z.string().min(1)).optional(),
  excludeGlobs: z.array(z.string().min(1)).optional(),
  languages: z.array(z.string().min(1)).optional(),
  maxFiles: z.number().int().min(1).max(10_000).optional(),
  maxChunks: z.number().int().min(1).max(100_000).optional(),
  maxTokens: z.number().int().min(1).max(10_000_000).optional(),
  maxChunkTokens: z.number().int().min(64).max(20_000).optional(),
  overlapTokens: z.number().int().min(0).max(5_000).optional(),
  force: z.boolean().optional(),
})

const RetrievalBudgetSchema = z.object({
  maxChunks: z.number().int().min(1),
  maxTokens: z.number().int().min(1),
})

export const RetrievalRequestSchema = z.object({
  projectId: z.string().min(1),
  query: z.string().min(1),
  maxChunks: z.number().int().min(1).optional(),
  maxTokens: z.number().int().min(1).optional(),
  budget: RetrievalBudgetSchema.optional(),
  dependencyExpansion: z.number().int().min(0).max(10).optional(),
})

type RetrievalRequest = z.infer<typeof RetrievalRequestSchema>

export type IndexingJobStatus = 'queued' | 'processing' | 'completed' | 'failed'

export interface IndexingJob {
  jobId: string
  request: IndexingRequest
  status: IndexingJobStatus
  attempts: number
  maxAttempts: number
  enqueuedAt: string
  updatedAt: string
  lastError?: string
  result?: unknown
}

export interface IndexingJobReceipt {
  jobId: string
  status: IndexingJobStatus
}

export interface IndexingJobQueue {
  enqueue(request: IndexingRequest): Promise<IndexingJobReceipt>
}

export interface CreateIndexerHonoAppOptions {
  queue?: IndexingJobQueue
}

type StoreWithStatusCounts = RepoIndexStore & {
  getIndexStatusCounts?: (projectId: string) => Promise<Record<string, number>>
}

export function createIndexerHonoApp(
  store: RepoIndexStore,
  options: CreateIndexerHonoAppOptions = {},
): Hono {
  const app = new Hono()
  const queue = options.queue ?? new FileIndexingQueue()

  app.get('/health', (context) => context.json({ ok: true }))

  app.post('/repo-indexes', async (context) => {
    const body = await context.req.json().catch(() => undefined)
    const parsed = IndexingRequestSchema.safeParse(body)

    if (!parsed.success) {
      return context.json({ error: 'invalid_request', issues: parsed.error.issues }, 400)
    }

    const receipt = await submitIndexingRequest(parsed.data, queue)
    return context.json(receipt, 202)
  })

  app.post('/retrieve', async (context) => {
    const body = await context.req.json().catch(() => undefined)
    const parsed = RetrievalRequestSchema.safeParse(body)

    if (!parsed.success) {
      return context.json({ error: 'invalid_request', issues: parsed.error.issues }, 400)
    }

    const queryStartedAt = performance.now()
    const result = await retrieveIndexedChunks(
      store,
      normalizeRetrievalRequest(parsed.data),
    )
    const elapsedMs = Math.round(performance.now() - queryStartedAt)

    return context.json({
      chunks: result.chunks,
      totalTokens: result.tokenCount,
      queryEmbeddingMs: elapsedMs,
      searchMs: 0,
    })
  })

  app.get('/symbols/:projectId', async (context) =>
    context.json(await store.listSymbols(context.req.param('projectId'))),
  )

  app.get('/graph/:projectId', async (context) => {
    const edges = await store.listDependencyEdges(context.req.param('projectId'))
    const nodes = new Set<string>()
    const graphEdges = edges.flatMap((edge) => {
      nodes.add(edge.sourceFilePath)

      const to = edge.resolvedFilePath || edge.importSpecifier

      if (!to) {
        return []
      }

      nodes.add(to)

      return [
        {
          from: edge.sourceFilePath,
          to,
          importedNames: edge.importedNames ?? [],
        },
      ]
    })

    return context.json({
      nodes: [...nodes].sort(),
      edges: graphEdges,
    })
  })

  app.get('/index/:projectId/status', async (context) => {
    const statusCounts = (store as StoreWithStatusCounts).getIndexStatusCounts

    if (!statusCounts) {
      return context.json({ status: 'unknown' })
    }

    return context.json({
      statuses: await statusCounts(context.req.param('projectId')),
    })
  })

  return app
}

function normalizeRetrievalRequest(input: RetrievalRequest): {
  projectId: string
  query: string
  budget?: RetrievalBudget
  dependencyExpansion?: number
} {
  const hasTopLevelBudget =
    input.maxChunks !== undefined || input.maxTokens !== undefined
  const budget = hasTopLevelBudget
    ? {
        maxChunks:
          input.maxChunks ?? input.budget?.maxChunks ?? defaultRetrievalBudget.maxChunks,
        maxTokens:
          input.maxTokens ?? input.budget?.maxTokens ?? defaultRetrievalBudget.maxTokens,
      }
    : input.budget

  return {
    projectId: input.projectId,
    query: input.query,
    budget,
    dependencyExpansion: input.dependencyExpansion,
  }
}

export class BullMQIndexingQueue implements IndexingJobQueue {
  readonly queue: Queue<IndexingRequest, unknown, string>
  private readonly connection: IORedis

  constructor(
    options: {
      queueName?: string
      redisUrl?: string
      maxAttempts?: number
    } = {},
  ) {
    this.connection = new IORedis(
      options.redisUrl ?? process.env.REDIS_URL ?? 'redis://127.0.0.1:6379',
      { maxRetriesPerRequest: null },
    )
    this.queue = new Queue<IndexingRequest, unknown, string>(
      options.queueName ?? 'repo-indexing',
      {
        connection: this.connection,
        defaultJobOptions: {
          attempts: options.maxAttempts ?? 3,
          backoff: { type: 'exponential', delay: 2_000 },
          removeOnComplete: 1_000,
          removeOnFail: 5_000,
        },
      },
    )
  }

  async enqueue(input: IndexingRequest): Promise<IndexingJobReceipt> {
    const request = parseIndexingRequest(input)
    const jobId = createIndexingJobId(request)
    const job = await this.queue.add('index-repository', request, { jobId })

    return {
      jobId: job.id ?? jobId,
      status: job.finishedOn ? 'completed' : 'queued',
    }
  }

  async close(): Promise<void> {
    await this.queue.close()
    this.connection.disconnect()
  }
}

export class FileIndexingQueue implements IndexingJobQueue {
  constructor(
    private readonly queueDirectory = defaultQueueDirectory(),
    private readonly maxAttempts = 3,
  ) {}

  async enqueue(input: IndexingRequest): Promise<IndexingJobReceipt> {
    const request = parseIndexingRequest(input)
    const jobId = createIndexingJobId(request)
    const existingStatus = await this.findExistingStatus(jobId)

    if (existingStatus) {
      return { jobId, status: existingStatus }
    }

    await this.ensureDirectories()

    const now = new Date().toISOString()
    const job: IndexingJob = {
      jobId,
      request,
      status: 'queued',
      attempts: 0,
      maxAttempts: this.maxAttempts,
      enqueuedAt: now,
      updatedAt: now,
    }

    await writeFile(
      this.jobPath('queued', jobId),
      JSON.stringify(job, null, 2),
      {
        flag: 'wx',
      },
    )

    return { jobId, status: 'queued' }
  }

  async claimNext(): Promise<IndexingJob | undefined> {
    await this.ensureDirectories()

    const queuedDirectory = this.statusDirectory('queued')
    const entries = await readdir(queuedDirectory)

    for (const entry of entries
      .filter((name) => name.endsWith('.json'))
      .sort()) {
      const jobId = entry.replace(/\.json$/u, '')
      const queuedPath = this.jobPath('queued', jobId)
      const processingPath = this.jobPath('processing', jobId)

      try {
        await rename(queuedPath, processingPath)
        const job = await this.readJob('processing', jobId)
        const updated: IndexingJob = {
          ...job,
          status: 'processing',
          attempts: job.attempts + 1,
          updatedAt: new Date().toISOString(),
        }
        await this.writeJob('processing', updated)
        return updated
      } catch (error) {
        if (!isFileSystemRace(error)) {
          throw error
        }
      }
    }

    return undefined
  }

  async complete(job: IndexingJob, result: unknown): Promise<void> {
    await this.moveFromProcessing(job, 'completed', { result })
  }

  async fail(job: IndexingJob, error: unknown): Promise<void> {
    const lastError = error instanceof Error ? error.message : String(error)

    if (job.attempts < job.maxAttempts) {
      await this.moveFromProcessing(job, 'queued', { lastError })
      return
    }

    await this.moveFromProcessing(job, 'failed', { lastError })
  }

  private async moveFromProcessing(
    job: IndexingJob,
    status: IndexingJobStatus,
    patch: Pick<IndexingJob, 'lastError' | 'result'>,
  ): Promise<void> {
    const updated: IndexingJob = {
      ...job,
      ...patch,
      status,
      updatedAt: new Date().toISOString(),
    }
    const processingPath = this.jobPath('processing', job.jobId)
    const targetPath = this.jobPath(status, job.jobId)

    await writeFile(processingPath, JSON.stringify(updated, null, 2))
    await rename(processingPath, targetPath)
  }

  private async readJob(
    status: IndexingJobStatus,
    jobId: string,
  ): Promise<IndexingJob> {
    return JSON.parse(await readFile(this.jobPath(status, jobId), 'utf8'))
  }

  private async writeJob(
    status: IndexingJobStatus,
    job: IndexingJob,
  ): Promise<void> {
    await writeFile(
      this.jobPath(status, job.jobId),
      JSON.stringify(job, null, 2),
    )
  }

  private async findExistingStatus(
    jobId: string,
  ): Promise<IndexingJobStatus | undefined> {
    for (const status of [
      'queued',
      'processing',
      'completed',
      'failed',
    ] satisfies IndexingJobStatus[]) {
      const exists = await stat(this.jobPath(status, jobId))
        .then(() => true)
        .catch(() => false)

      if (exists) {
        return status
      }
    }

    return undefined
  }

  private async ensureDirectories(): Promise<void> {
    await Promise.all(
      (
        [
          'queued',
          'processing',
          'completed',
          'failed',
        ] satisfies IndexingJobStatus[]
      ).map((status) =>
        mkdir(this.statusDirectory(status), { recursive: true }),
      ),
    )
  }

  private statusDirectory(status: IndexingJobStatus): string {
    return path.join(this.queueDirectory, status)
  }

  private jobPath(status: IndexingJobStatus, jobId: string): string {
    return path.join(
      this.statusDirectory(status),
      `${safeFileName(jobId)}.json`,
    )
  }
}

export async function submitIndexingRequest(
  body: unknown,
  queue: IndexingJobQueue,
): Promise<IndexingJobReceipt> {
  return queue.enqueue(parseIndexingRequest(body))
}

export function parseIndexingRequest(body: unknown): IndexingRequest {
  return IndexingRequestSchema.parse(body)
}

export function createIndexingJobId(request: IndexingRequest): string {
  if (request.requestId) {
    return safeFileName(request.requestId)
  }

  return createHash('sha256')
    .update(
      JSON.stringify({
        projectId: request.projectId,
        repoPath: path.resolve(request.repoPath),
        workflowId: request.workflowId,
        includeGlobs: request.includeGlobs ?? [],
        force: request.force ?? false,
      }),
    )
    .digest('hex')
}

function defaultQueueDirectory(): string {
  return path.resolve(
    process.env.REPO_INDEXER_QUEUE_DIR ?? '.traycer/repo-index-queue',
  )
}

function safeFileName(input: string): string {
  return input.replace(/[^A-Za-z0-9_.-]/gu, '_')
}

function isFileSystemRace(error: unknown): boolean {
  return (
    error instanceof Error &&
    'code' in error &&
    ['ENOENT', 'EEXIST'].includes(String(error.code))
  )
}
