import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { VideoCacheService } from './video-cache.service';

@Module({
    imports: [ConfigModule],
    providers: [VideoCacheService],
    exports: [VideoCacheService],
})
export class CacheModule { }
