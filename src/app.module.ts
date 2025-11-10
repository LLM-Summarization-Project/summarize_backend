import { Module } from '@nestjs/common';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { SummarizeModule } from './summarize/summarize.module';
import { QueueModule } from './queue/queue.module';

@Module({
  imports: [SummarizeModule, QueueModule,],
  controllers: [AppController],
  providers: [AppService],
})
export class AppModule {}
