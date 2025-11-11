import { Module } from '@nestjs/common';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { SummarizeModule } from './summarize/summarize.module';
import { QueueModule } from './queue/queue.module';
import { ProgressModule } from './summarize/progress.module';

@Module({
  imports: [SummarizeModule, QueueModule, ProgressModule],
  controllers: [AppController],
  providers: [AppService],
})
export class AppModule {}
