import { Worker } from 'bullmq';
import { processor } from './processor';

let worker: Worker | null = null;

export async function startWorker(concurrency: number) {
  if (worker) {
    console.log('Closing old worker...');
    await worker.close();
  }

  console.log('Starting worker with concurrency =', concurrency);

  worker = new Worker(
    'summarize',
    processor, // <<< logic ยาว ๆ ของคุณ
    {
      concurrency,
      connection: {
        host: process.env.REDIS_HOST ?? 'localhost',
        port: Number(process.env.REDIS_PORT ?? 6379),
      },
    },
  );
}

export function getWorker() {
  return worker;
}
