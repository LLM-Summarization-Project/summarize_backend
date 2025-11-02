import { PartialType } from '@nestjs/mapped-types';
import { SummarizeRequestDto } from './create-summarize.dto';

export class UpdateSummarizeDto extends PartialType(SummarizeRequestDto) {}
