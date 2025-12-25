import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import Redis from 'ioredis';

export interface CachedSummary {
    id: string;
    youtubeUrl: string;
    status: string;
}

@Injectable()
export class VideoCacheService {
    private readonly redis: Redis;
    private readonly logger = new Logger(VideoCacheService.name);

    // TTL: 1 month in seconds (30 days * 24 hours * 60 minutes * 60 seconds)
    private readonly CACHE_TTL = 2592000;
    private readonly CACHE_PREFIX = 'summary:video:';

    constructor(private configService: ConfigService) {
        this.redis = new Redis({
            host: this.configService.get<string>('REDIS_HOST') ?? 'localhost',
            port: Number(this.configService.get<number>('REDIS_PORT') ?? 6379),
        });
    }

    /**
     * Extract YouTube video ID from various URL formats
     * Handles: youtube.com/watch?v=, youtu.be/, youtube.com/embed/
     */
    extractVideoId(url: string): string | null {
        const patterns = [
            /(?:youtube\.com\/watch\?v=)([a-zA-Z0-9_-]{11})/,
            /(?:youtu\.be\/)([a-zA-Z0-9_-]{11})/,
            /(?:youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})/,
            /(?:youtube\.com\/v\/)([a-zA-Z0-9_-]{11})/,
            /(?:youtube\.com\/shorts\/)([a-zA-Z0-9_-]{11})/,  // YouTube Shorts
        ];

        for (const pattern of patterns) {
            const match = url.match(pattern);
            if (match) return match[1];
        }

        return null;
    }

    /**
     * Generate cache key from YouTube URL
     * Falls back to base64 encoding if video ID cannot be extracted
     */
    getCacheKey(youtubeUrl: string): string {
        const videoId = this.extractVideoId(youtubeUrl);
        if (videoId) {
            return `${this.CACHE_PREFIX}${videoId}`;
        }
        // Fallback: use base64 of URL
        return `${this.CACHE_PREFIX}${Buffer.from(youtubeUrl).toString('base64').slice(0, 32)}`;
    }

    /**
     * Check if summary exists in cache
     * Returns cached data or null if not found
     */
    async getCachedSummary(youtubeUrl: string): Promise<CachedSummary | null> {
        try {
            const cacheKey = this.getCacheKey(youtubeUrl);
            const cached = await this.redis.get(cacheKey);

            if (cached) {
                this.logger.log(`Cache HIT for ${cacheKey}`);
                return JSON.parse(cached);
            }

            this.logger.debug(`Cache MISS for ${cacheKey}`);
            return null;
        } catch (error) {
            this.logger.error(`Cache read error: ${error.message}`);
            return null; // Graceful fallback - don't block on cache errors
        }
    }

    /**
     * Store completed summary in cache
     */
    async setCachedSummary(youtubeUrl: string, data: CachedSummary): Promise<void> {
        try {
            const cacheKey = this.getCacheKey(youtubeUrl);
            await this.redis.setex(cacheKey, this.CACHE_TTL, JSON.stringify(data));
            this.logger.log(`Cached summary for ${cacheKey} (TTL: 1 month)`);
        } catch (error) {
            this.logger.error(`Cache write error: ${error.message}`);
            // Don't throw - caching failure shouldn't break the main flow
        }
    }

    /**
     * Manually invalidate cache for a video
     */
    async invalidateCache(youtubeUrl: string): Promise<void> {
        try {
            const cacheKey = this.getCacheKey(youtubeUrl);
            await this.redis.del(cacheKey);
            this.logger.log(`Invalidated cache for ${cacheKey}`);
        } catch (error) {
            this.logger.error(`Cache invalidation error: ${error.message}`);
        }
    }
}
