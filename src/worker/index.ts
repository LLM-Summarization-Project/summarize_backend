import { startWorker, updateConcurrency } from "./worker-manager";
import { getConcurrency, subscribeConcurrencyChanges, getRedisConfig } from "../shared/redis-concurrency";
import Redis from 'ioredis';
import 'dotenv/config';

async function bootstrap() {
    // Get initial concurrency from Redis
    const redis = new Redis(getRedisConfig());
    const initialConcurrency = await getConcurrency(redis);
    await redis.quit();
    
    console.log(`[Worker] Starting with concurrency: ${initialConcurrency}`);
    await startWorker(initialConcurrency);
    
    // Subscribe to concurrency changes
    subscribeConcurrencyChanges((newValue) => {
        console.log(`[Worker] Received concurrency update: ${newValue}`);
        updateConcurrency(newValue);
    });
    
    console.log('[Worker] Ready and listening for concurrency changes');
}

bootstrap();