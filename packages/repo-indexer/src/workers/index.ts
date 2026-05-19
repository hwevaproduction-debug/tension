import { setTimeout as delay } from 'node:timers/promises'

import { isCancellationRequested } from '@traycer/queue'
import { Worker, type Job } from 'bullmq'
import IORedis from 'ioredis'
import OpenAI from 'openai'

import type { FileIndexingQueue, IndexingJob } from '../api'
import { indexRepository } from '../indexer'
import { buildFileSummaryPrompt, deterministicSummaryModel } from '../summarizer'
import type { IndexingResult, RepoIndexStore } from '../types'

export interface ProcessIndexingJobOptions {
  store: RepoIndexStore
  isCancelled?: (workflowId: string) => Promise<boolean>
}

export interface ProcessNextIndexingJobOptions extends ProcessIndexingJobOptions {
  queue: FileIndexingQueue
}

export interface RunIndexingWorkerOptions extends ProcessNextIndexingJobOptions {
  pollIntervalMs?: number
  once?: boolean
  signal?: AbortSignal
  onResult?: (result: IndexingResult) => void
}

export interface RunBullMQIndexingWorkerOptions extends ProcessIndexingJobOptions {
  queueName?: string
  redisUrl?: string
  concurrency?: number
}

export interface CreateBullMQSummaryWorkerOptions {
  store: RepoIndexStore
  queueName?: string
  redisUrl?: string
  concurrency?: number
}

export interface SummaryJobData {
  fileId: string
}

type StoreWithSummaryContext = RepoIndexStore & {
  getSummaryContextForFile?: (fileId: string) => Promise<{
    filePath: string
    symbols: { name: string | null; isExported: boolean | null }[]
    imports: { importSpecifier: string | null }[]
    deterministicSummary?: { summary: string | null; model: string | null } | null
  } | undefined>
}

export async function processIndexingJob(
  job: IndexingJob,
  options: ProcessIndexingJobOptions,
): Promise<IndexingResult> {
  const workflowId = job.request.workflowId ?? job.jobId
  const isCancelled =
    options.isCancelled ?? ((id: string) => isCancellationRequested(id))

  if (await isCancelled(workflowId)) {
    return {
      projectId: job.request.projectId,
      repoPath: job.request.repoPath,
      filesSeen: 0,
      filesIndexed: 0,
      filesSkipped: 0,
      chunksIndexed: 0,
      tokenCount: 0,
      budgetExhausted: false,
      files: [],
    }
  }

  return indexRepository(job.request, options.store)
}

export async function processNextIndexingJob(
  options: ProcessNextIndexingJobOptions,
): Promise<IndexingResult | undefined> {
  const job = await options.queue.claimNext()

  if (!job) {
    return undefined
  }

  try {
    const result = await processIndexingJob(job, options)
    await options.queue.complete(job, result)
    return result
  } catch (error) {
    await options.queue.fail(job, error)
    return undefined
  }
}

export async function runIndexingWorker(
  options: RunIndexingWorkerOptions,
): Promise<void> {
  const pollIntervalMs = options.pollIntervalMs ?? 1_000

  while (!options.signal?.aborted) {
    const result = await processNextIndexingJob(options)

    if (result) {
      options.onResult?.(result)
    }

    if (options.once) {
      return
    }

    await delay(pollIntervalMs, undefined, { signal: options.signal }).catch(
      (error) => {
        if (error instanceof Error && error.name === 'AbortError') {
          return
        }

        throw error
      },
    )
  }
}

export function createBullMQIndexingWorker(
  options: RunBullMQIndexingWorkerOptions,
): Worker {
  const connection = new IORedis(
    options.redisUrl ?? process.env.REDIS_URL ?? 'redis://127.0.0.1:6379',
    { maxRetriesPerRequest: null },
  )

  const worker = new Worker(
    options.queueName ?? 'repo-indexing',
    async (job: Job<IndexingJob['request']>) => {
      await job.updateProgress({ phase: 'indexing', filesIndexed: 0 })
      const result = await processIndexingJob(
        {
          jobId: job.id ?? String(job.name),
          request: job.data,
          status: 'processing',
          attempts: job.attemptsMade + 1,
          maxAttempts: job.opts.attempts ?? 3,
          enqueuedAt: new Date(job.timestamp).toISOString(),
          updatedAt: new Date().toISOString(),
        },
        options,
      )
      await job.updateProgress({
        phase: 'completed',
        filesIndexed: result.filesIndexed,
        filesSkipped: result.filesSkipped,
        chunksIndexed: result.chunksIndexed,
      })
      return result
    },
    {
      connection,
      concurrency: options.concurrency ?? 2,
      lockDuration: 120_000,
      stalledInterval: 30_000,
    },
  )

  worker.on('closed', () => connection.disconnect())
  worker.on('failed', (job, error) => {
    process.stderr.write(
      `repo-indexing job ${job?.id ?? 'unknown'} failed: ${
        error instanceof Error ? error.message : String(error)
      }\n`,
    )
  })

  return worker
}

export function createBullMQSummaryWorker(
  options: CreateBullMQSummaryWorkerOptions,
): Worker {
  const connection = new IORedis(
    options.redisUrl ?? process.env.REDIS_URL ?? 'redis://127.0.0.1:6379',
    { maxRetriesPerRequest: null },
  )

  const worker = new Worker(
    options.queueName ?? 'repo:summarize',
    async (job: Job<SummaryJobData>) => {
      try {
        const context = await (options.store as StoreWithSummaryContext)
          .getSummaryContextForFile?.(job.data.fileId)
        const filePath = context?.filePath ?? job.data.fileId
        const prompt = buildFileSummaryPrompt({
          filePath,
          symbols: (context?.symbols ?? []).map((symbol) => ({
            name: symbol.name ?? '',
            isExported: symbol.isExported ?? false,
          })),
          imports: (context?.imports ?? []).map((item) => ({
            importSpecifier: item.importSpecifier ?? '',
          })),
        })
        const fallback = normalizeDeterministicSummary(context?.deterministicSummary)
        const summaryResult = process.env.OPENAI_API_KEY
          ? await retrySummary(() => summarizeWithOpenAI(prompt)).catch((error) => {
              process.stderr.write(
                `repo-summary job ${job.id ?? 'unknown'} OpenAI summary failed: ${
                  error instanceof Error ? error.message : String(error)
                }\n`,
              )
              if (!fallback) {
                throw error
              }

              return fallback
            })
          : fallback

        if (!summaryResult) {
          process.stderr.write(
            `repo-summary job ${job.id ?? 'unknown'} skipped persistence: deterministic summary fallback is unavailable\n`,
          )
          return { ok: false, fileId: job.data.fileId }
        }

        if (options.store.persistSummary) {
          await options.store
            .persistSummary(job.data.fileId, summaryResult.summary, summaryResult.model)
            .catch((error) => {
              process.stderr.write(
                `repo-summary job ${job.id ?? 'unknown'} persist failed: ${
                  error instanceof Error ? error.message : String(error)
                }\n`,
              )
            })
        } else {
          process.stderr.write(
            `repo-summary job ${job.id ?? 'unknown'} skipped persistence: store.persistSummary is unavailable\n`,
          )
        }

        return { ok: true, fileId: job.data.fileId, model: summaryResult.model }
      } catch (error) {
        process.stderr.write(
          `repo-summary job ${job.id ?? 'unknown'} failed non-blocking: ${
            error instanceof Error ? error.message : String(error)
          }\n`,
        )
        return { ok: false, fileId: job.data.fileId }
      }
    },
    {
      connection,
      concurrency: options.concurrency ?? 2,
      lockDuration: 120_000,
      stalledInterval: 30_000,
    },
  )

  worker.on('closed', () => connection.disconnect())
  worker.on('failed', (job, error) => {
    process.stderr.write(
      `repo-summary job ${job?.id ?? 'unknown'} failed: ${
        error instanceof Error ? error.message : String(error)
      }\n`,
    )
  })

  return worker
}

function normalizeDeterministicSummary(
  summary: { summary: string | null; model: string | null } | null | undefined,
): { summary: string; model: string } | undefined {
  if (!summary?.summary) {
    return undefined
  }

  return {
    summary: summary.summary,
    model: summary.model ?? deterministicSummaryModel,
  }
}

async function summarizeWithOpenAI(prompt: string): Promise<{ summary: string; model: string }> {
  const model = 'gpt-4o-mini'
  const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY })
  const response = await client.chat.completions.create({
    model,
    messages: [
      {
        role: 'user',
        content: prompt,
      },
    ],
  })

  return {
    summary: response.choices[0]?.message.content?.trim() || prompt,
    model,
  }
}

async function retrySummary<T>(operation: () => Promise<T>): Promise<T> {
  let lastError: unknown

  for (let attempt = 1; attempt <= 2; attempt += 1) {
    try {
      return await operation()
    } catch (error) {
      lastError = error

      if (attempt < 2) {
        await delay(2_000 * 2 ** (attempt - 1))
      }
    }
  }

  throw lastError
}
