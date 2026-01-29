import { IsNumber, IsString } from "class-validator";
import { ApiProperty } from "@nestjs/swagger";

export class CreateChatDto {
    @ApiProperty()
    @IsString()
    message: string;

    @ApiProperty()
    @IsString()
    summaryId: string;

    @ApiProperty()
    @IsNumber()
    userId: number;

    @ApiProperty({ required: false, enum: ['transcript', 'description'] })
    @IsString()
    contextType?: 'transcript' | 'description';

    @ApiProperty({ required: false })
    @IsString()
    topicId?: string;

    @ApiProperty({ required: false })
    @IsString()
    customContext?: string;
}
