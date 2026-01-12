import Redis from 'ioredis';

const CONCURRENCY_KEY = 'system:concurrency';
const CONCURRENCY_CHANNEL = 'concurrency:changed';
const DEFAULT_CONCURRENCY = 2;

export function getRedisConfig() {
  return {
    host: process.env.REDIS_HOST ?? 'localhost',
    port: Number(process.env.REDIS_PORT ?? 6379),
  };
}

export async function getConcurrency(redis: Redis): Promise<number> {
  const value = await redis.get(CONCURRENCY_KEY);
  return value ? parseInt(value, 10) : DEFAULT_CONCURRENCY;
}

export async function setConcurrency(redis: Redis, value: number): Promise<void> {
  await redis.set(CONCURRENCY_KEY, value.toString());
  await redis.publish(CONCURRENCY_CHANNEL, value.toString());
  console.log(`[Redis] Concurrency set to ${value} and published`);
}

export function subscribeConcurrencyChanges(
  onConcurrencyChange: (newValue: number) => void
): Redis {
  const subscriber = new Redis(getRedisConfig());
  
  subscriber.subscribe(CONCURRENCY_CHANNEL, (err) => {
    if (err) {
      console.error('[Redis] Failed to subscribe to concurrency channel:', err);
    } else {
      console.log(`[Redis] Subscribed to ${CONCURRENCY_CHANNEL}`);
    }
  });

  subscriber.on('message', (channel, message) => {
    if (channel === CONCURRENCY_CHANNEL) {
      const newValue = parseInt(message, 10);
      if (!isNaN(newValue)) {
        console.log(`[Redis] Received concurrency change: ${newValue}`);
        onConcurrencyChange(newValue);
      }
    }
  });

  return subscriber;
}
