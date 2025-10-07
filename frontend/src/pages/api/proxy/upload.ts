import { NextApiRequest, NextApiResponse } from 'next';
import httpProxy from 'http-proxy';

// Disable Next.js body parsing to stream the request
export const config = {
  api: {
    bodyParser: false,
  },
};

const proxy = httpProxy.createProxyServer();

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  return new Promise<void>((resolve, reject) => {
    // Remove 'host' header to avoid conflicts
    delete req.headers.host;

    // Rewrite the URL to include the correct path and a trailing slash
    req.url = '/api/upload/';

    proxy.web(
      req,
      res,
      {
        target: 'http://localhost:8000', // Your backend server
        changeOrigin: true,
        selfHandleResponse: false,
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