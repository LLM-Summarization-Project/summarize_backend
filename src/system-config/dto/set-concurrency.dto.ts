import {IsInt, Min, Max} from 'class-validator';

export class SetConcurrencyDto {
    @IsInt()
    @Min(1)
    @Max(5)
    value: number;
}