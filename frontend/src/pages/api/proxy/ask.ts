import { NextApiRequest, NextApiResponse } from 'next';
import httpProxy from 'http-proxy';
import { Readable } from 'stream';

const proxy = httpProxy.createProxyServer();

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  return new Promise<void>((resolve, reject) => {
    // The request body has already been parsed by Next.js,
    // so we need to re-serialize it to forward to the target.
    const body = JSON.stringify(req.body);
    req.headers['Content-Length'] = Buffer.byteLength(body).toString();
    req.headers['Content-Type'] = 'application/json';

    // Rewrite the URL to include the correct path and a trailing slash
    req.url = '/api/ask/';

    proxy.web(
      req,
      res,
      {
        target: 'http://localhost:8000', // Your backend server
        changeOrigin: true,
        selfHandleResponse: false,
        // We need to stream the body again
        ignorePath: false,
        buffer: Readable.from(Buffer.from(body)),
      },
      (err) => {
        if (err) {
          console.error('Proxy error:', err);
          res.status(500).json({ message: 'Proxy error' });
          return reject(err);
        }
        resolve();
      }
    );
  });
}