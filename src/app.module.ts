import { Module } from '@nestjs/common';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { SummarizeModule } from './summarize/summarize.module';

@Module({
  imports: [SummarizeModule],
  controllers: [AppController],
  providers: [AppService],
})
export class AppModule {}
