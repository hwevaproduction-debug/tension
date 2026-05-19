export type FileIndexStatus = 'pending' | 'indexed' | 'skipped' | 'failed'

export interface IndexingRequest {
  projectId: string
  repoPath: string
  workflowId?: string
  requestId?: string
  includeGlobs?: string[]
  excludeGlobs?: string[]
  languages?: string[]
  maxFiles?: number
  maxChunks?: number
  maxTokens?: number
  maxChunkTokens?: number
  overlapTokens?: number
  force?: boolean
}

export interface ParsedFile {
  projectId: string
  repoPath: string
  absolutePath: string
  filePath: string
  language: string
  content: string
  contentHash: string
  lineCount: number
}

export interface RepoChunk {
  chunkIndex: number
  chunkType: 'file' | 'symbol'
  symbolName?: string
  content: string
  startLine: number
  endLine: number
  tokenCount: number
}

export interface EmbeddedRepoChunk extends RepoChunk {
  embedding: number[]
}

export interface ExtractedSymbol {
  name: string
  kind: string
  isExported: boolean
  isDefaultExport: boolean
  line: number
  endLine?: number
}

export interface ExtractedImport {
  importSpecifier: string
  isExternal: boolean
  importedNames: string[]
}

export interface FileSummary {
  summary: string
  model: string
}

export interface StoredIndexedFile {
  id: string
  projectId: string
  repoPath: string
  filePath: string
  language: string
  contentHash: string
  lineCount: number | null
  indexedAt: Date | null
  status: FileIndexStatus
}

export interface StoredRepoChunk {
  id: string
  fileId: string
  filePath: string
  chunkIndex: number
  chunkType: string | null
  symbolName: string | null
  content: string
  startLine: number | null
  endLine: number | null
  tokenCount: number
  embedding?: number[] | null
}

export interface StoredRepoSymbol {
  id: string
  fileId: string
  filePath: string
  name: string | null
  kind: string | null
  isExported: boolean | null
  isDefaultExport: boolean | null
  chunkId: string | null
}

export interface StoredDependencyEdge {
  sourceFileId: string
  sourceFilePath: string
  resolvedFileId: string | null
  resolvedFilePath: string | null
  importSpecifier: string | null
  isExternal: boolean | null
  importedNames: string[] | null
}

export interface PersistIndexedFileInput {
  file: ParsedFile
  chunks: EmbeddedRepoChunk[]
  symbols: ExtractedSymbol[]
  imports: ExtractedImport[]
  summary: FileSummary
}

export interface PersistIndexedFileResult {
  fileId: string
  chunksIndexed: number
}

export interface RepoIndexStore {
  getFileByPath(input: {
    projectId: string
    repoPath: string
    filePath: string
  }): Promise<StoredIndexedFile | undefined>
  persistIndexedFile(
    input: PersistIndexedFileInput,
  ): Promise<PersistIndexedFileResult>
  listChunks(projectId: string): Promise<StoredRepoChunk[]>
  listSymbols(projectId: string): Promise<StoredRepoSymbol[]>
  listDependencyEdges(projectId: string): Promise<StoredDependencyEdge[]>
  semanticSearchChunks?(
    projectId: string,
    embedding: number[],
    limit: number,
  ): Promise<RetrievalCandidate[]>
  persistSummary?(
    fileId: string,
    summary: string,
    model: string,
  ): Promise<void>
}

export interface IndexedFileResult {
  filePath: string
  status: FileIndexStatus
  contentHash: string
  chunksIndexed: number
  reason?: string
}

export interface IndexingResult {
  projectId: string
  repoPath: string
  filesSeen: number
  filesIndexed: number
  filesSkipped: number
  chunksIndexed: number
  tokenCount: number
  budgetExhausted: boolean
  files: IndexedFileResult[]
}

export interface RetrievalBudget {
  maxChunks: number
  maxTokens: number
}

export interface RetrievalCandidate {
  chunk: StoredRepoChunk
  score: number
}

export interface RetrievalResult {
  chunks: StoredRepoChunk[]
  tokenCount: number
  budget: RetrievalBudget
}
