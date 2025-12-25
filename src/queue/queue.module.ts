import { Global, Module } from '@nestjs/common';
import { QueueService } from './queue.service';
import { QueueEventsListener } from './queue.event';
import { ProgressModule } from 'src/summarize/progress.module';
import { QueueController } from './queue.controller';
import { SystemConfigModule } from 'src/system-config/system-config.module';

@Global()
@Module({
  imports: [ProgressModule, SystemConfigModule],
  providers: [QueueService, QueueEventsListener],
  controllers: [QueueController],
  exports: [QueueService],
})
export class QueueModule {}
