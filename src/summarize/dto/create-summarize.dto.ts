import { ApiProperty, ApiPropertyOptional } from "@nestjs/swagger";

export class SummarizeRequestDto {
    @ApiProperty({
        example: 'https://youtu.be/Rq7plosixd0?si=-Xw05o5mTZd-eTBt',
        description: 'ลิงก์ Youtube ที่ต้องการสรุปเนื้อหา'
    })
    youtubeUrl: string;

    @ApiPropertyOptional({
        example: 0.0,
        description: 'Whisper temperature (0.0-1.0). ถ้าไม่ส่งจะใช้ค่าจาก environment'
    })
    whisperTemp?: number;

    @ApiPropertyOptional({
        example: false,
        description: 'ใช้ YouTube Transcript API (true/false). if not set, will be false'
    })
    youtubeApi?: boolean;
}

export type SummaryStatus = 'QUEUED' | 'RUNNING' | 'DONE' | 'ERROR' | 'CANCEL';

export interface SummaryResult {
    transcript_path?: string;
    bullet_points_path?: string;
    articles_path?: string;
    scene_facts_path?: string;
}

export interface SummaryState {
    status: SummaryStatus;
    step?: string;
    percent?: number;
    result?: SummaryResult;
    errorMessage?: string;
    startedAt?: Date;
    finishedAt?: Date;
}