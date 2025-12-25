import { startWorker } from "./worker-manager";
import 'dotenv/config';

async function bootstrap() {
    await startWorker(Number(process.env.BULL_CONCURRENCY ?? 2))
}

bootstrap();