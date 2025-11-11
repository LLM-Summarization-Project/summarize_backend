import { Global, Module } from '@nestjs/common';
import { QueueService } from './queue.service';
import { QueueEventsListener } from './queue.event';
import { ProgressModule } from 'src/summarize/progress.module';

@Global()
@Module({
  imports: [ProgressModule],
  providers: [QueueService, QueueEventsListener],
  exports: [QueueService],
})
export class QueueModule {}
