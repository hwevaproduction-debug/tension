import { createHash } from 'node:crypto'
import { readFile, readdir, stat } from 'node:fs/promises'
import path from 'node:path'

import ignore from 'ignore'

import type { IndexingRequest, ParsedFile } from '../types'

const skippedDirectories = new Set([
  '.git',
  '.next',
  '.turbo',
  'build',
  'coverage',
  'dist',
  'logs',
  'node_modules',
  'target',
  'vendor',
])

const extensionLanguage = new Map<string, string>([
  ['.c', 'c'],
  ['.cc', 'cpp'],
  ['.cpp', 'cpp'],
  ['.cs', 'csharp'],
  ['.css', 'css'],
  ['.go', 'go'],
  ['.h', 'c'],
  ['.hpp', 'cpp'],
  ['.html', 'html'],
  ['.java', 'java'],
  ['.js', 'javascript'],
  ['.jsx', 'javascript'],
  ['.json', 'json'],
  ['.kt', 'kotlin'],
  ['.md', 'markdown'],
  ['.mjs', 'javascript'],
  ['.py', 'python'],
  ['.rs', 'rust'],
  ['.sh', 'shell'],
  ['.sql', 'sql'],
  ['.ts', 'typescript'],
  ['.tsx', 'typescript'],
  ['.yaml', 'yaml'],
  ['.yml', 'yaml'],
])

const maxReadableFileBytes = 1_000_000

export async function parseRepository(
  request: IndexingRequest,
): Promise<ParsedFile[]> {
  const repoPath = path.resolve(request.repoPath)
  const files = await discoverRepositoryFiles(repoPath, {
    includeGlobs: request.includeGlobs,
    excludeGlobs: request.excludeGlobs,
    languages: request.languages,
    maxFiles: request.maxFiles ?? 2_000,
  })
  const parsedFiles: ParsedFile[] = []

  for (const absolutePath of files) {
    const fileStat = await stat(absolutePath)

    if (fileStat.size > maxReadableFileBytes) {
      continue
    }

    const content = await readFile(absolutePath, 'utf8')

    if (content.includes('\0')) {
      continue
    }

    const filePath = toRepoRelativePath(repoPath, absolutePath)

    parsedFiles.push({
      projectId: request.projectId,
      repoPath,
      absolutePath,
      filePath,
      language: detectLanguage(filePath),
      content,
      contentHash: hashContent(content),
      lineCount: content.length === 0 ? 0 : content.split(/\r?\n/u).length,
    })
  }

  return parsedFiles
}

export async function discoverRepositoryFiles(
  repoPath: string,
  options: {
    includeGlobs?: string[]
    excludeGlobs?: string[]
    languages?: string[]
    maxFiles: number
  },
): Promise<string[]> {
  const root = path.resolve(repoPath)
  const discovered: string[] = []
  const includeGlobs = options.includeGlobs ?? []
  const excludeMatcher = ignore().add(options.excludeGlobs ?? [])
  const languageFilter = new Set(options.languages ?? [])
  const rootIgnore = ignore().add(await readGitignore(path.join(root, '.gitignore')))

  async function visit(directoryPath: string, inheritedIgnores: string[]): Promise<void> {
    if (discovered.length >= options.maxFiles) {
      return
    }

    const relativeDirectory = toRepoRelativePath(root, directoryPath)
    const localGitignore = await readGitignore(path.join(directoryPath, '.gitignore'))
    const localPrefix = relativeDirectory ? `${relativeDirectory}/` : ''
    const activeIgnores = [
      ...inheritedIgnores,
      ...localGitignore.map((pattern) => `${localPrefix}${pattern}`),
    ]
    const matcher = ignore().add(activeIgnores)
    const entries = await readdir(directoryPath, { withFileTypes: true })

    for (const entry of entries.sort((left, right) =>
      left.name.localeCompare(right.name),
    )) {
      if (discovered.length >= options.maxFiles) {
        return
      }

      const absolutePath = path.join(directoryPath, entry.name)
      const relativePath = toRepoRelativePath(root, absolutePath)

      if (rootIgnore.ignores(relativePath) || matcher.ignores(relativePath)) {
        continue
      }

      if (entry.isDirectory()) {
        if (!skippedDirectories.has(entry.name)) {
          await visit(absolutePath, activeIgnores)
        }
        continue
      }

      if (!entry.isFile()) {
        continue
      }

      const language = detectLanguage(relativePath)
      if (
        languageFilter.size > 0 &&
        !languageFilter.has(language) &&
        !languageFilter.has(path.extname(relativePath).slice(1))
      ) {
        continue
      }

      if (
        extensionLanguage.has(path.extname(relativePath).toLowerCase()) &&
        matchesIncludeGlobs(relativePath, includeGlobs) &&
        !excludeMatcher.ignores(relativePath)
      ) {
        discovered.push(absolutePath)
      }
    }
  }

  await visit(root, await readGitignore(path.join(root, '.gitignore')))

  return discovered
}

export function hashContent(content: string): string {
  return createHash('sha256').update(content).digest('hex')
}

export function detectLanguage(filePath: string): string {
  return extensionLanguage.get(path.extname(filePath).toLowerCase()) ?? 'text'
}

async function readGitignore(filePath: string): Promise<string[]> {
  return readFile(filePath, 'utf8')
    .then((content) =>
      content
        .split(/\r?\n/u)
        .map((line) => line.trim())
        .filter((line) => line && !line.startsWith('#')),
    )
    .catch(() => [])
}

function toRepoRelativePath(repoPath: string, absolutePath: string): string {
  const relativePath = path.relative(repoPath, absolutePath).replaceAll(path.sep, '/')
  return relativePath === '' ? '' : relativePath
}

function matchesIncludeGlobs(relativePath: string, includeGlobs: string[]): boolean {
  if (includeGlobs.length === 0) {
    return true
  }

  return includeGlobs.some((pattern) => globToRegExp(pattern).test(relativePath))
}

function globToRegExp(pattern: string): RegExp {
  let source = '^'

  for (let index = 0; index < pattern.length; index += 1) {
    const character = pattern[index]
    const next = pattern[index + 1]

    if (character === '*') {
      if (next === '*') {
        const afterGlobstar = pattern[index + 2]
        index += 1
        source += afterGlobstar === '/' ? '(?:.*/)?' : '.*'
        if (afterGlobstar === '/') {
          index += 1
        }
      } else {
        source += '[^/]*'
      }
      continue
    }

    if (character === '?') {
      source += '[^/]'
      continue
    }

    source += escapeRegExp(character)
  }

  return new RegExp(`${source}$`, 'u')
}

function escapeRegExp(input: string): string {
  return input.replace(/[\\^$+?.()|[\]{}]/gu, '\\$&')
}
