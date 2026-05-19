import { estimateTokenCount } from '../chunker'
import type { ExtractedImport, ExtractedSymbol, FileSummary, ParsedFile } from '../types'

export const deterministicSummaryModel = 'deterministic-local-summary-v1'

export function summarizeFile(file: ParsedFile, maxTokens = 120): FileSummary {
  const meaningfulLines = file.content
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)

  const firstLine = meaningfulLines[0] ?? `${file.filePath} is empty.`
  let summary = `${file.filePath}: ${firstLine}`

  while (estimateTokenCount(summary) > maxTokens && summary.length > 20) {
    summary = summary.slice(0, Math.floor(summary.length * 0.8)).trim()
  }

  return {
    summary,
    model: deterministicSummaryModel,
  }
}

export function buildFileSummaryPrompt(
  input: {
    filePath: string
    symbols: Pick<ExtractedSymbol, 'name' | 'isExported'>[]
    imports: Pick<ExtractedImport, 'importSpecifier'>[]
  },
  maxTokens = 400,
): string {
  const exportedSymbolNames = input.symbols
    .filter((symbol) => symbol.isExported && symbol.name)
    .map((symbol) => symbol.name)
    .join(', ')
  const importSpecifiers = input.imports
    .map((item) => item.importSpecifier)
    .filter(Boolean)
    .join(', ')
  let prompt = [
    `Summarize this file: ${input.filePath}`,
    `Exports: ${exportedSymbolNames}`,
    `Imports: ${importSpecifiers}`,
  ].join('\n')

  while (estimateTokenCount(prompt) > maxTokens && prompt.length > 20) {
    prompt = prompt.slice(0, Math.floor(prompt.length * 0.8)).trim()
  }

  return prompt
}
