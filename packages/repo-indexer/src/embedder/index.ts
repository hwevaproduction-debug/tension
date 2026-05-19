import { createHash } from 'node:crypto'

import OpenAI from 'openai'

import type { EmbeddedRepoChunk, RepoChunk } from '../types'

export const embeddingDimensions = 1536
export const defaultEmbeddingModel = 'text-embedding-3-small'

class LruCache<K, V> {
  private readonly entries = new Map<K, V>()

  constructor(private readonly maxSize = 2_000) {}

  get(key: K): V | undefined {
    const value = this.entries.get(key)

    if (value !== undefined) {
      this.entries.delete(key)
      this.entries.set(key, value)
    }

    return value
  }

  set(key: K, value: V): void {
    if (this.entries.has(key)) {
      this.entries.delete(key)
    } else if (this.entries.size >= this.maxSize) {
      const oldestKey = this.entries.keys().next().value

      if (oldestKey !== undefined) {
        this.entries.delete(oldestKey)
      }
    }

    this.entries.set(key, value)
  }
}

const embeddingCache = new LruCache<string, number[]>()

export interface EmbeddingProvider {
  readonly model: string
  embed(input: string[]): Promise<number[][]>
}

export class OpenAIEmbeddingProvider implements EmbeddingProvider {
  readonly model: string
  private readonly client: OpenAI

  constructor(options: { apiKey?: string; model?: string } = {}) {
    const apiKey = options.apiKey ?? process.env.OPENAI_API_KEY

    if (!apiKey) {
      throw new Error('OPENAI_API_KEY is required for OpenAI embeddings')
    }

    this.model = options.model ?? process.env.REPO_INDEXER_EMBEDDING_MODEL ?? defaultEmbeddingModel
    this.client = new OpenAI({ apiKey })
  }

  async embed(input: string[]): Promise<number[][]> {
    if (input.length === 0) {
      return []
    }

    const response = await this.client.embeddings.create({
      model: this.model,
      input,
      dimensions: embeddingDimensions,
    })

    return response.data
      .sort((left, right) => left.index - right.index)
      .map((item) => normalizeEmbedding(item.embedding))
  }
}

export class HashingEmbeddingProvider implements EmbeddingProvider {
  readonly model = 'local-feature-hashing-1536-v1'

  async embed(input: string[]): Promise<number[][]> {
    return input.map(createDeterministicEmbedding)
  }
}

export function createEmbeddingProvider(): EmbeddingProvider {
  return process.env.OPENAI_API_KEY
    ? new OpenAIEmbeddingProvider()
    : new HashingEmbeddingProvider()
}

export async function embedChunks(
  chunks: RepoChunk[],
  provider: EmbeddingProvider = createEmbeddingProvider(),
): Promise<EmbeddedRepoChunk[]> {
  const embeddings = await embedTexts(
    chunks.map((chunk) => chunk.content),
    provider,
  )

  return chunks.map((chunk, index) => ({
    ...chunk,
    embedding: embeddings[index] ?? createDeterministicEmbedding(chunk.content),
  }))
}

export async function embedTexts(
  inputs: string[],
  provider: EmbeddingProvider = createEmbeddingProvider(),
): Promise<number[][]> {
  const results = new Array<number[]>(inputs.length)
  const misses: string[] = []
  const missIndexes: number[] = []

  inputs.forEach((input, index) => {
    const key = cacheKey(provider.model, input)
    const cached = embeddingCache.get(key)

    if (cached) {
      results[index] = cached
      return
    }

    misses.push(input)
    missIndexes.push(index)
  })

  const batchSize = 64
  for (let offset = 0; offset < misses.length; offset += batchSize) {
    const batch = misses.slice(offset, offset + batchSize)
    const batchIndexes = missIndexes.slice(offset, offset + batchSize)
    const embeddings = await provider.embed(batch)

    embeddings.forEach((embedding, index) => {
      const normalized = normalizeEmbedding(embedding)
      const resultIndex = batchIndexes[index]
      results[resultIndex] = normalized
      embeddingCache.set(cacheKey(provider.model, inputs[resultIndex]), normalized)
    })
  }

  return results
}

export function createDeterministicEmbedding(input: string): number[] {
  const vector = new Array<number>(embeddingDimensions).fill(0)
  const tokens = input.toLowerCase().match(/[a-z0-9_$]+/gu) ?? []

  for (const token of tokens) {
    const seed = createHash('sha256').update(token).digest()
    const index = seed.readUInt16BE(0) % embeddingDimensions
    const sign = seed[2] % 2 === 0 ? 1 : -1
    vector[index] += sign * Math.log(1 + token.length)
  }

  return normalizeEmbedding(vector)
}

export function cosineSimilarity(left: number[], right: number[]): number {
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

function normalizeEmbedding(embedding: number[]): number[] {
  const resized =
    embedding.length === embeddingDimensions
      ? [...embedding]
      : Array.from({ length: embeddingDimensions }, (_, index) => embedding[index] ?? 0)
  const magnitude = Math.sqrt(resized.reduce((sum, value) => sum + value ** 2, 0))

  if (magnitude === 0) {
    return resized
  }

  return resized.map((value) => Number((value / magnitude).toFixed(8)))
}

function cacheKey(model: string, input: string): string {
  return `${model}:${createHash('sha256').update(input).digest('hex')}`
}
