import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { SummarizeModule } from './summarize/summarize.module';
import { QueueModule } from './queue/queue.module';
import { ProgressModule } from './summarize/progress.module';
import { AuthModule } from './auth/auth.module';
import { SystemConfigModule } from './system-config/system-config.module';
import { ChatModule } from './chat/chat.module';

@Module({
  imports: [
    ConfigModule.forRoot({ isGlobal: true }),
    AuthModule, 
    SummarizeModule, 
    QueueModule, 
    ProgressModule, SystemConfigModule, ChatModule
  ],
  controllers: [AppController],
  providers: [AppService],
})
export class AppModule {}
