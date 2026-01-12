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
    processor,
    {
      concurrency,
      connection: {
        host: process.env.REDIS_HOST ?? 'localhost',
        port: Number(process.env.REDIS_PORT ?? 6379),
      },
    },
  );
}

/**
 * Update worker concurrency dynamically without restarting
 */
export function updateConcurrency(newValue: number) {
  if (worker) {
    worker.concurrency = newValue;
    console.log(`[Worker] Concurrency updated to ${newValue}`);
  } else {
    console.warn('[Worker] Cannot update concurrency: worker not initialized');
  }
}

export function getWorker() {
  return worker;
}
