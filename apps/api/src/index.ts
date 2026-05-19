import {
  createServer,
  type IncomingMessage,
  type Server,
  type ServerResponse,
} from 'node:http'
import { pathToFileURL } from 'node:url'

import {
  createDrizzleRepoIndexStore,
  createIndexerHonoApp,
  createRepoIndexerDatabase,
  FileIndexingQueue,
  InMemoryRepoIndexStore,
  type RepoIndexStore,
} from '@traycer/repo-indexer'

export interface ApiRuntimeOptions {
  port?: number
  host?: string
  queue?: FileIndexingQueue
  store?: RepoIndexStore
}

export interface ApiRuntime {
  name: '@traycer/api'
  role: 'http-api'
  start(): Promise<{ host: string; port: number }>
  close(): Promise<void>
}

export function createApiRuntime(options: ApiRuntimeOptions = {}): ApiRuntime {
  const host = options.host ?? process.env.API_HOST ?? '127.0.0.1'
  const port =
    options.port ?? Number.parseInt(process.env.API_PORT ?? '3001', 10)
  const queue = options.queue ?? new FileIndexingQueue()
  const databaseRuntime = options.store
    ? undefined
    : process.env.DATABASE_URL
      ? createRepoIndexerDatabase()
      : undefined
  const store =
    options.store ??
    (databaseRuntime
      ? createDrizzleRepoIndexStore(databaseRuntime.db)
      : new InMemoryRepoIndexStore())
  const app = createIndexerHonoApp(store, { queue })
  const server = createServer((request, response) => {
    handleRequest(request, response, app).catch((error) => {
      writeJson(response, 500, {
        error: error instanceof Error ? error.message : String(error),
      })
    })
  })

  return {
    name: '@traycer/api',
    role: 'http-api',
    start() {
      return new Promise((resolve, reject) => {
        server.once('error', reject)
        server.listen(port, host, () => {
          const address = server.address()
          const listeningPort =
            typeof address === 'object' && address ? address.port : port

          server.off('error', reject)
          resolve({ host, port: listeningPort })
        })
      })
    },
    async close() {
      await closeServer(server)
      await databaseRuntime?.close()
    },
  }
}

export const apiRuntime = createApiRuntime()

async function handleRequest(
  request: IncomingMessage,
  response: ServerResponse,
  app: ReturnType<typeof createIndexerHonoApp>,
): Promise<void> {
  const body = request.method === 'GET' || request.method === 'HEAD'
    ? undefined
    : await readBody(request)
  const requestBody = body ? toArrayBuffer(body) : undefined
  const honoResponse = await app.fetch(
    new Request(`http://${request.headers.host ?? '127.0.0.1'}${request.url ?? '/'}`, {
      method: request.method,
      headers: requestHeaders(request),
      body: requestBody,
    }),
  )

  response.writeHead(
    honoResponse.status,
    Object.fromEntries(honoResponse.headers.entries()),
  )

  if (honoResponse.body) {
    const reader = honoResponse.body.getReader()

    while (true) {
      const { done, value } = await reader.read()

      if (done) {
        break
      }

      response.write(Buffer.from(value))
    }
  }

  response.end()
}

function writeJson(
  response: ServerResponse,
  statusCode: number,
  body: unknown,
): void {
  response.writeHead(statusCode, { 'content-type': 'application/json' })
  response.end(JSON.stringify(body))
}

function closeServer(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.close((error) => {
      if (error) {
        reject(error)
        return
      }

      resolve()
    })
  })
}

async function readBody(request: IncomingMessage): Promise<Buffer> {
  const chunks: Buffer[] = []

  for await (const chunk of request) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk))
  }

  return Buffer.concat(chunks)
}

function requestHeaders(request: IncomingMessage): Headers {
  const headers = new Headers()

  for (const [name, value] of Object.entries(request.headers)) {
    if (Array.isArray(value)) {
      value.forEach((item) => headers.append(name, item))
    } else if (value !== undefined) {
      headers.set(name, value)
    }
  }

  return headers
}

function toArrayBuffer(buffer: Buffer): ArrayBuffer {
  const arrayBuffer = new ArrayBuffer(buffer.length)
  new Uint8Array(arrayBuffer).set(buffer)
  return arrayBuffer
}

if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(process.argv[1]).href
) {
  apiRuntime
    .start()
    .then(({ host, port }) => {
      process.stdout.write(`api listening on http://${host}:${port}\n`)
    })
    .catch((error) => {
      process.stderr.write(
        `${error instanceof Error ? error.stack : String(error)}\n`,
      )
      process.exitCode = 1
    })
}
