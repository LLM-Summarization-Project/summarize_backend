import { IsNumber, IsString } from "class-validator";

export class CreateChatDto {
    @IsString()
    message: string;

    @IsString()
    summaryId: string;

    @IsNumber()
    userId: number;
}
