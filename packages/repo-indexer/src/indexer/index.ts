import { randomUUID } from 'node:crypto'
import path from 'node:path'

import { and, desc, eq, sql } from 'drizzle-orm'

import { chunkParsedFile } from '../chunker'
import { createDeterministicEmbedding, createEmbeddingProvider, embedChunks } from '../embedder'
import { extractFileMetadata } from '../extractor'
import { parseRepository } from '../parser'
import { selectChunksWithinBudget } from '../retrieval'
import { deterministicSummaryModel, summarizeFile } from '../summarizer'
import {
  repoIndexChunks,
  repoIndexFiles,
  repoIndexImports,
  repoIndexSummaries,
  repoIndexSymbols,
} from '../db/schema'
import type { RepoIndexerDbClient } from '../db'
import type {
  EmbeddedRepoChunk,
  FileSummary,
  FileIndexStatus,
  IndexedFileResult,
  IndexingRequest,
  IndexingResult,
  PersistIndexedFileInput,
  PersistIndexedFileResult,
  RepoIndexStore,
  StoredDependencyEdge,
  StoredIndexedFile,
  StoredRepoChunk,
  StoredRepoSymbol,
} from '../types'

const defaultIndexingBudget = {
  maxChunks: 1_000,
  maxTokens: 200_000,
}

export async function indexRepository(
  request: IndexingRequest,
  store: RepoIndexStore,
): Promise<IndexingResult> {
  const parsedFiles = await parseRepository(request)
  const embeddingProvider = createEmbeddingProvider()
  const files: IndexedFileResult[] = []
  let filesIndexed = 0
  let filesSkipped = 0
  let chunksIndexed = 0
  let tokenCount = 0
  let budgetExhausted = false
  const maxChunks = request.maxChunks ?? defaultIndexingBudget.maxChunks
  const maxTokens = request.maxTokens ?? defaultIndexingBudget.maxTokens

  for (const file of parsedFiles) {
    const existing = await store.getFileByPath({
      projectId: request.projectId,
      repoPath: file.repoPath,
      filePath: file.filePath,
    })

    if (
      !request.force &&
      existing?.contentHash === file.contentHash &&
      existing.status === 'indexed'
    ) {
      filesSkipped += 1
      files.push({
        filePath: file.filePath,
        status: 'skipped',
        contentHash: file.contentHash,
        chunksIndexed: 0,
        reason: 'unchanged',
      })
      continue
    }

    if (chunksIndexed >= maxChunks || tokenCount >= maxTokens) {
      budgetExhausted = true
      filesSkipped += 1
      files.push({
        filePath: file.filePath,
        status: 'skipped',
        contentHash: file.contentHash,
        chunksIndexed: 0,
        reason: 'indexing-budget-exhausted',
      })
      continue
    }

    const metadata = extractFileMetadata(file)
    const selectedChunks = selectChunksWithinBudget(
      chunkParsedFile(file, {
        maxChunkTokens: request.maxChunkTokens ?? 800,
        overlapTokens: request.overlapTokens ?? 80,
      }, metadata.symbols).map((chunk) => ({
        chunk: {
          id: `${file.filePath}:${chunk.chunkIndex}`,
          fileId: file.filePath,
          filePath: file.filePath,
          chunkIndex: chunk.chunkIndex,
          chunkType: chunk.chunkType,
          symbolName: chunk.symbolName ?? null,
          content: chunk.content,
          startLine: chunk.startLine,
          endLine: chunk.endLine,
          tokenCount: chunk.tokenCount,
        },
        score: 1,
      })),
      {
        maxChunks: maxChunks - chunksIndexed,
        maxTokens: maxTokens - tokenCount,
      },
    ).chunks

    if (selectedChunks.length === 0) {
      budgetExhausted = true
      filesSkipped += 1
      files.push({
        filePath: file.filePath,
        status: 'skipped',
        contentHash: file.contentHash,
        chunksIndexed: 0,
        reason: 'file-exceeds-token-budget',
      })
      continue
    }

    const chunks = await embedChunks(
      selectedChunks.map((chunk) => ({
        chunkIndex: chunk.chunkIndex,
        chunkType: chunk.chunkType === 'symbol' ? 'symbol' : 'file',
        symbolName: chunk.symbolName ?? undefined,
        content: chunk.content,
        startLine: chunk.startLine ?? 1,
        endLine: chunk.endLine ?? 1,
        tokenCount: chunk.tokenCount,
      })),
      embeddingProvider,
    )
    const persisted = await store.persistIndexedFile({
      file,
      chunks,
      symbols: metadata.symbols,
      imports: metadata.imports,
      summary: summarizeFile(file),
    })

    filesIndexed += 1
    chunksIndexed += persisted.chunksIndexed
    tokenCount += chunks.reduce((sum, chunk) => sum + chunk.tokenCount, 0)
    files.push({
      filePath: file.filePath,
      status: 'indexed',
      contentHash: file.contentHash,
      chunksIndexed: persisted.chunksIndexed,
    })
  }

  return {
    projectId: request.projectId,
    repoPath: parsedFiles[0]?.repoPath ?? request.repoPath,
    filesSeen: parsedFiles.length,
    filesIndexed,
    filesSkipped,
    chunksIndexed,
    tokenCount,
    budgetExhausted,
    files,
  }
}

export class InMemoryRepoIndexStore implements RepoIndexStore {
  private readonly files = new Map<string, StoredIndexedFile>()
  private readonly chunks = new Map<string, StoredRepoChunk[]>()
  private readonly symbols = new Map<string, StoredRepoSymbol[]>()
  private readonly imports = new Map<string, StoredDependencyEdge[]>()
  private readonly summaries = new Map<string, { summary: string; model: string }>()
  private readonly deterministicSummaries = new Map<
    string,
    { summary: string; model: string }
  >()

  async getFileByPath(input: {
    projectId: string
    repoPath: string
    filePath: string
  }): Promise<StoredIndexedFile | undefined> {
    return this.files.get(fileKey(input))
  }

  async persistIndexedFile(
    input: PersistIndexedFileInput,
  ): Promise<PersistIndexedFileResult> {
    const key = fileKey(input.file)
    const existing = this.files.get(key)
    const fileId = existing?.id ?? randomUUID()
    const indexedAt = new Date()

    this.files.set(key, {
      id: fileId,
      projectId: input.file.projectId,
      repoPath: input.file.repoPath,
      filePath: input.file.filePath,
      language: input.file.language,
      contentHash: input.file.contentHash,
      lineCount: input.file.lineCount,
      indexedAt,
      status: 'indexed',
    })

    this.chunks.set(
      fileId,
      input.chunks.map((chunk) => ({
        id: randomUUID(),
        fileId,
        filePath: input.file.filePath,
        chunkIndex: chunk.chunkIndex,
        chunkType: chunk.chunkType,
        symbolName: chunk.symbolName ?? null,
        content: chunk.content,
        startLine: chunk.startLine,
        endLine: chunk.endLine,
        tokenCount: chunk.tokenCount,
        embedding: chunk.embedding,
      })),
    )
    this.symbols.set(
      fileId,
      input.symbols.map((symbol) => ({
        id: randomUUID(),
        fileId,
        filePath: input.file.filePath,
        name: symbol.name,
        kind: symbol.kind,
        isExported: symbol.isExported,
        isDefaultExport: symbol.isDefaultExport,
        chunkId: null,
      })),
    )
    this.imports.set(
      fileId,
      input.imports.map((item) => ({
        sourceFileId: fileId,
        sourceFilePath: input.file.filePath,
        resolvedFileId: null,
        resolvedFilePath: resolveImportPath(input.file.filePath, item.importSpecifier),
        importSpecifier: item.importSpecifier,
        isExternal: item.isExternal,
        importedNames: item.importedNames,
      })),
    )
    this.deterministicSummaries.set(fileId, input.summary)
    this.summaries.set(fileId, input.summary)

    return {
      fileId,
      chunksIndexed: input.chunks.length,
    }
  }

  async listChunks(projectId: string): Promise<StoredRepoChunk[]> {
    const fileIds = [...this.files.values()]
      .filter((file) => file.projectId === projectId)
      .map((file) => file.id)
    const chunks = fileIds.flatMap((fileId) => this.chunks.get(fileId) ?? [])

    return chunks.sort((left, right) => {
      if (left.filePath !== right.filePath) {
        return left.filePath.localeCompare(right.filePath)
      }

      return left.chunkIndex - right.chunkIndex
    })
  }

  async listSymbols(projectId: string): Promise<StoredRepoSymbol[]> {
    const fileIds = [...this.files.values()]
      .filter((file) => file.projectId === projectId)
      .map((file) => file.id)

    return fileIds.flatMap((fileId) => this.symbols.get(fileId) ?? [])
  }

  async listDependencyEdges(projectId: string): Promise<StoredDependencyEdge[]> {
    const files = [...this.files.values()].filter(
      (file) => file.projectId === projectId,
    )
    const byPath = new Map(files.map((file) => [file.filePath, file]))

    return files.flatMap((file) =>
      (this.imports.get(file.id) ?? []).map((edge) => {
        const resolvedPath = importPathCandidates(
          edge.sourceFilePath,
          edge.importSpecifier ?? '',
        ).find((candidate) => byPath.has(candidate))
        const resolvedFile = resolvedPath ? byPath.get(resolvedPath) : undefined
        return {
          ...edge,
          resolvedFileId: resolvedFile?.id ?? null,
          resolvedFilePath: resolvedFile?.filePath ?? edge.resolvedFilePath,
        }
      }),
    )
  }

  async semanticSearchChunks(
    projectId: string,
    embedding: number[],
    limit: number,
  ) {
    const chunks = await this.listChunks(projectId)

    return chunks
      .map((chunk) => ({
        chunk,
        score: chunk.embedding
          ? cosineSimilarityLocal(embedding, chunk.embedding)
          : 0,
      }))
      .sort((left, right) => right.score - left.score)
      .slice(0, limit)
  }

  async persistSummary(
    fileId: string,
    summary: string,
    model: string,
  ): Promise<void> {
    this.summaries.set(fileId, { summary, model })
  }

  async getSummaryContextForFile(fileId: string): Promise<FileSummaryContext | undefined> {
    const file = [...this.files.values()].find((item) => item.id === fileId)

    if (!file) {
      return undefined
    }

    return {
      filePath: file.filePath,
      symbols: this.symbols.get(fileId) ?? [],
      imports: this.imports.get(fileId) ?? [],
      deterministicSummary: this.deterministicSummaries.get(fileId),
    }
  }
}

export function createDrizzleRepoIndexStore(
  db: RepoIndexerDbClient,
): RepoIndexStore & RepoIndexStoreWithSummaryContext & RepoIndexStoreWithStatusCounts {
  return {
    async getFileByPath(input) {
      const rows = await db
        .select()
        .from(repoIndexFiles)
        .where(
          and(
            eq(repoIndexFiles.projectId, input.projectId),
            eq(repoIndexFiles.repoPath, input.repoPath),
            eq(repoIndexFiles.filePath, input.filePath),
          ),
        )
        .orderBy(desc(repoIndexFiles.updatedAt))
        .limit(1)

      return rows[0] ? toStoredIndexedFile(rows[0]) : undefined
    },

    async persistIndexedFile(input) {
      const now = new Date()
      const existingRows = await db
        .select()
        .from(repoIndexFiles)
        .where(
          and(
            eq(repoIndexFiles.projectId, input.file.projectId),
            eq(repoIndexFiles.repoPath, input.file.repoPath),
            eq(repoIndexFiles.filePath, input.file.filePath),
          ),
        )
        .orderBy(desc(repoIndexFiles.updatedAt))
        .limit(1)
      const existing = existingRows[0]
        ? toStoredIndexedFile(existingRows[0])
        : undefined
      const fileId =
        existing?.id ??
        (
          await db
            .insert(repoIndexFiles)
            .values({
              projectId: input.file.projectId,
              repoPath: input.file.repoPath,
              filePath: input.file.filePath,
              language: input.file.language,
              contentHash: input.file.contentHash,
              lineCount: input.file.lineCount,
              indexedAt: now,
              status: 'indexed',
              updatedAt: now,
            })
            .returning({ id: repoIndexFiles.id })
        )[0].id

      if (existing) {
        await db
          .update(repoIndexFiles)
          .set({
            language: input.file.language,
            contentHash: input.file.contentHash,
            lineCount: input.file.lineCount,
            indexedAt: now,
            status: 'indexed',
            updatedAt: now,
          })
          .where(eq(repoIndexFiles.id, fileId))
      }

      await replaceFileMetadata(db, fileId, input)

      return {
        fileId,
        chunksIndexed: input.chunks.length,
      }
    },

    async listChunks(projectId) {
      const rows = await db
        .select({
          id: repoIndexChunks.id,
          fileId: repoIndexChunks.fileId,
          filePath: repoIndexFiles.filePath,
          chunkIndex: repoIndexChunks.chunkIndex,
          chunkType: repoIndexChunks.chunkType,
          symbolName: repoIndexChunks.symbolName,
          content: repoIndexChunks.content,
          startLine: repoIndexChunks.startLine,
          endLine: repoIndexChunks.endLine,
          tokenCount: repoIndexChunks.tokenCount,
          embedding: repoIndexChunks.embedding,
        })
        .from(repoIndexChunks)
        .innerJoin(
          repoIndexFiles,
          eq(repoIndexChunks.fileId, repoIndexFiles.id),
        )
        .where(eq(repoIndexFiles.projectId, projectId))

      return rows
        .map((row) => ({
          ...row,
          chunkIndex: row.chunkIndex ?? 0,
          tokenCount: row.tokenCount ?? 0,
        }))
        .sort((left, right) => {
          if (left.filePath !== right.filePath) {
            return left.filePath.localeCompare(right.filePath)
          }

          return left.chunkIndex - right.chunkIndex
        })
    },

    async listSymbols(projectId) {
      const rows = await db
        .select({
          id: repoIndexSymbols.id,
          fileId: repoIndexSymbols.fileId,
          filePath: repoIndexFiles.filePath,
          name: repoIndexSymbols.name,
          kind: repoIndexSymbols.kind,
          isExported: repoIndexSymbols.isExported,
          isDefaultExport: repoIndexSymbols.isDefaultExport,
          chunkId: repoIndexSymbols.chunkId,
        })
        .from(repoIndexSymbols)
        .innerJoin(repoIndexFiles, eq(repoIndexSymbols.fileId, repoIndexFiles.id))
        .where(eq(repoIndexFiles.projectId, projectId))

      return rows
    },

    async listDependencyEdges(projectId) {
      const rows = await db
        .select({
          sourceFileId: repoIndexImports.sourceFileId,
          sourceFilePath: repoIndexFiles.filePath,
          resolvedFileId: repoIndexImports.resolvedFileId,
          importSpecifier: repoIndexImports.importSpecifier,
          isExternal: repoIndexImports.isExternal,
          importedNames: repoIndexImports.importedNames,
        })
        .from(repoIndexImports)
        .innerJoin(
          repoIndexFiles,
          eq(repoIndexImports.sourceFileId, repoIndexFiles.id),
        )
        .where(eq(repoIndexFiles.projectId, projectId))

      const projectFiles = await db
        .select({
          id: repoIndexFiles.id,
          filePath: repoIndexFiles.filePath,
        })
        .from(repoIndexFiles)
        .where(eq(repoIndexFiles.projectId, projectId))
      const filePathById = new Map(projectFiles.map((file) => [file.id, file.filePath]))

      return rows.map((row) => ({
        ...row,
        resolvedFilePath: row.resolvedFileId
          ? filePathById.get(row.resolvedFileId) ?? null
          : null,
      }))
    },

    async semanticSearchChunks(projectId, embedding, limit) {
      const vector = `[${embedding.join(',')}]`
      const distance = sql<number>`${repoIndexChunks.embedding} <=> ${vector}::vector`
      const rows = await db
        .select({
          id: repoIndexChunks.id,
          fileId: repoIndexChunks.fileId,
          filePath: repoIndexFiles.filePath,
          chunkIndex: repoIndexChunks.chunkIndex,
          chunkType: repoIndexChunks.chunkType,
          symbolName: repoIndexChunks.symbolName,
          content: repoIndexChunks.content,
          startLine: repoIndexChunks.startLine,
          endLine: repoIndexChunks.endLine,
          tokenCount: repoIndexChunks.tokenCount,
          embedding: repoIndexChunks.embedding,
          distance,
        })
        .from(repoIndexChunks)
        .innerJoin(
          repoIndexFiles,
          eq(repoIndexChunks.fileId, repoIndexFiles.id),
        )
        .where(eq(repoIndexFiles.projectId, projectId))
        .orderBy(distance)
        .limit(limit)

      return rows.map((row) => ({
        chunk: {
          id: row.id,
          fileId: row.fileId,
          filePath: row.filePath,
          chunkIndex: row.chunkIndex ?? 0,
          chunkType: row.chunkType,
          symbolName: row.symbolName,
          content: row.content,
          startLine: row.startLine,
          endLine: row.endLine,
          tokenCount: row.tokenCount ?? 0,
          embedding: row.embedding,
        },
        score: 1 - Number(row.distance ?? 1),
      }))
    },

    async persistSummary(fileId, summary, model) {
      await db.insert(repoIndexSummaries).values({
        fileId,
        summary,
        summaryEmbedding: createDeterministicEmbedding(summary),
        model,
      })
    },

    async getSummaryContextForFile(fileId) {
      const files = await db
        .select({
          id: repoIndexFiles.id,
          filePath: repoIndexFiles.filePath,
          projectId: repoIndexFiles.projectId,
        })
        .from(repoIndexFiles)
        .where(eq(repoIndexFiles.id, fileId))
        .limit(1)
      const file = files[0]

      if (!file) {
        return undefined
      }

      const [symbols, imports, deterministicSummaries] = await Promise.all([
        db
          .select({
            id: repoIndexSymbols.id,
            fileId: repoIndexSymbols.fileId,
            filePath: repoIndexFiles.filePath,
            name: repoIndexSymbols.name,
            kind: repoIndexSymbols.kind,
            isExported: repoIndexSymbols.isExported,
            isDefaultExport: repoIndexSymbols.isDefaultExport,
            chunkId: repoIndexSymbols.chunkId,
          })
          .from(repoIndexSymbols)
          .innerJoin(repoIndexFiles, eq(repoIndexSymbols.fileId, repoIndexFiles.id))
          .where(eq(repoIndexSymbols.fileId, fileId)),
        db
          .select({
            sourceFileId: repoIndexImports.sourceFileId,
            sourceFilePath: repoIndexFiles.filePath,
            resolvedFileId: repoIndexImports.resolvedFileId,
            resolvedFilePath: sql<string | null>`null`,
            importSpecifier: repoIndexImports.importSpecifier,
            isExternal: repoIndexImports.isExternal,
            importedNames: repoIndexImports.importedNames,
          })
          .from(repoIndexImports)
          .innerJoin(
            repoIndexFiles,
            eq(repoIndexImports.sourceFileId, repoIndexFiles.id),
          )
          .where(eq(repoIndexImports.sourceFileId, fileId)),
        db
          .select({
            summary: repoIndexSummaries.summary,
            model: repoIndexSummaries.model,
          })
          .from(repoIndexSummaries)
          .where(
            and(
              eq(repoIndexSummaries.fileId, fileId),
              eq(repoIndexSummaries.model, deterministicSummaryModel),
            ),
          )
          .orderBy(desc(repoIndexSummaries.createdAt))
          .limit(1),
      ])
      const deterministicSummary = deterministicSummaries[0]?.summary
        ? {
            summary: deterministicSummaries[0].summary,
            model: deterministicSummaries[0].model ?? deterministicSummaryModel,
          }
        : undefined

      return {
        filePath: file.filePath,
        symbols,
        imports,
        deterministicSummary,
      }
    },

    async getIndexStatusCounts(projectId) {
      const rows = await db
        .select({
          status: repoIndexFiles.status,
          count: sql<number>`count(*)`,
        })
        .from(repoIndexFiles)
        .where(eq(repoIndexFiles.projectId, projectId))
        .groupBy(repoIndexFiles.status)

      return Object.fromEntries(
        rows.map((row) => [row.status, Number(row.count)]),
      )
    },
  }
}

export interface FileSummaryContext {
  filePath: string
  symbols: StoredRepoSymbol[]
  imports: StoredDependencyEdge[]
  deterministicSummary?: FileSummary
}

export interface RepoIndexStoreWithSummaryContext extends RepoIndexStore {
  getSummaryContextForFile(fileId: string): Promise<FileSummaryContext | undefined>
}

export interface RepoIndexStoreWithStatusCounts extends RepoIndexStore {
  getIndexStatusCounts(projectId: string): Promise<Record<string, number>>
}

async function replaceFileMetadata(
  db: RepoIndexerDbClient,
  fileId: string,
  input: PersistIndexedFileInput,
): Promise<void> {
  await db
    .delete(repoIndexImports)
    .where(eq(repoIndexImports.sourceFileId, fileId))
  await db.delete(repoIndexSymbols).where(eq(repoIndexSymbols.fileId, fileId))
  await db
    .delete(repoIndexSummaries)
    .where(eq(repoIndexSummaries.fileId, fileId))
  await db.delete(repoIndexChunks).where(eq(repoIndexChunks.fileId, fileId))

  const insertedChunks =
    input.chunks.length === 0
      ? []
      : await db
          .insert(repoIndexChunks)
          .values(
            input.chunks.map((chunk) => ({
              fileId,
              chunkIndex: chunk.chunkIndex,
              chunkType: chunk.chunkType,
              symbolName: chunk.symbolName,
              content: chunk.content,
              startLine: chunk.startLine,
              endLine: chunk.endLine,
              tokenCount: chunk.tokenCount,
              embedding: chunk.embedding,
            })),
          )
          .returning({
            id: repoIndexChunks.id,
            chunkIndex: repoIndexChunks.chunkIndex,
          })

  if (input.symbols.length > 0) {
    await db.insert(repoIndexSymbols).values(
      input.symbols.map((symbol) => ({
        fileId,
        name: symbol.name,
        kind: symbol.kind,
        isExported: symbol.isExported,
        isDefaultExport: symbol.isDefaultExport,
        chunkId: findChunkIdForLine(insertedChunks, input.chunks, symbol.line),
      })),
    )
  }

  if (input.imports.length > 0) {
    const projectFiles = await db
      .select({
        id: repoIndexFiles.id,
        filePath: repoIndexFiles.filePath,
      })
      .from(repoIndexFiles)
      .where(
        and(
          eq(repoIndexFiles.projectId, input.file.projectId),
          eq(repoIndexFiles.repoPath, input.file.repoPath),
        ),
      )
    const fileIdByPath = new Map(projectFiles.map((file) => [file.filePath, file.id]))

    await db.insert(repoIndexImports).values(
      input.imports.map((item) => ({
        sourceFileId: fileId,
        importSpecifier: item.importSpecifier,
        resolvedFileId: findResolvedFileId(
          fileIdByPath,
          input.file.filePath,
          item.importSpecifier,
        ),
        isExternal: item.isExternal,
        importedNames: item.importedNames,
      })),
    )
  }

  await db.insert(repoIndexSummaries).values({
    fileId,
    summary: input.summary.summary,
    summaryEmbedding: createDeterministicEmbedding(input.summary.summary),
    model: input.summary.model,
  })
}

function findChunkIdForLine(
  insertedChunks: { id: string; chunkIndex: number | null }[],
  chunks: EmbeddedRepoChunk[],
  line: number,
): string | null {
  const chunk = chunks.find(
    (item) => item.startLine <= line && item.endLine >= line,
  )
  const insertedChunk = insertedChunks.find(
    (item) => item.chunkIndex === chunk?.chunkIndex,
  )

  return insertedChunk?.id ?? null
}

function toStoredIndexedFile(
  row: typeof repoIndexFiles.$inferSelect,
): StoredIndexedFile {
  return {
    id: row.id,
    projectId: row.projectId,
    repoPath: row.repoPath,
    filePath: row.filePath,
    language: row.language,
    contentHash: row.contentHash,
    lineCount: row.lineCount,
    indexedAt: row.indexedAt,
    status: row.status as FileIndexStatus,
  }
}

function fileKey(input: {
  projectId: string
  repoPath: string
  filePath: string
}): string {
  return `${input.projectId}:${input.repoPath}:${input.filePath}`
}

function resolveImportPath(
  sourceFilePath: string,
  importSpecifier: string,
): string | null {
  return importPathCandidates(sourceFilePath, importSpecifier)[0] ?? null
}

function findResolvedFileId(
  fileIdByPath: Map<string, string>,
  sourceFilePath: string,
  importSpecifier: string,
): string | null {
  for (const candidate of importPathCandidates(sourceFilePath, importSpecifier)) {
    const fileId = fileIdByPath.get(candidate)
    if (fileId) {
      return fileId
    }
  }

  return null
}

function importPathCandidates(
  sourceFilePath: string,
  importSpecifier: string,
): string[] {
  if (!importSpecifier.startsWith('.')) {
    return []
  }

  const sourceDirectory = path.posix.dirname(sourceFilePath)
  const basePath = path.posix.normalize(path.posix.join(sourceDirectory, importSpecifier))
  const extension = path.posix.extname(basePath)
  return extension
    ? [basePath]
    : [
        basePath,
        `${basePath}.ts`,
        `${basePath}.tsx`,
        `${basePath}.js`,
        `${basePath}.jsx`,
        path.posix.join(basePath, 'index.ts'),
        path.posix.join(basePath, 'index.tsx'),
        path.posix.join(basePath, 'index.js'),
        path.posix.join(basePath, 'index.jsx'),
      ]
}

function cosineSimilarityLocal(left: number[], right: number[]): number {
  let dot = 0
  let leftMagnitude = 0
  let rightMagnitude = 0

  for (let index = 0; index < Math.min(left.length, right.length); index += 1) {
    dot += left[index] * right[index]
    leftMagnitude += left[index] ** 2
    rightMagnitude += right[index] ** 2
  }

  if (leftMagnitude === 0 || rightMagnitude === 0) {
    return 0
  }

  return dot / (Math.sqrt(leftMagnitude) * Math.sqrt(rightMagnitude))
}
