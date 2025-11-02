import { Controller, Get, Post, Body, Patch, Param, Delete } from '@nestjs/common';
import { SummarizeService } from './summarize.service';
import { SummarizeRequestDto } from './dto/create-summarize.dto';
import { UpdateSummarizeDto } from './dto/update-summarize.dto';
import { ApiTags } from '@nestjs/swagger';

@ApiTags('summary')
@Controller('summarize')
export class SummarizeController {
  constructor(private readonly summarizeService: SummarizeService) {}

  @Post()
  createSummary(@Body() summarizeRequestDto: SummarizeRequestDto) {
    return this.summarizeService.createSummary(summarizeRequestDto.youtubeUrl);
  }

  @Get(':id')
  getSummary(@Param('id') id: string) {
    return this.summarizeService.getSummary(id)
  }

  // @Get()
  // findAll() {
  //   return this.summarizeService.findAll();
  // }

  // @Get(':id')
  // findOne(@Param('id') id: string) {
  //   return this.summarizeService.findOne(+id);
  // }

  // @Patch(':id')
  // update(@Param('id') id: string, @Body() updateSummarizeDto: UpdateSummarizeDto) {
  //   return this.summarizeService.update(+id, updateSummarizeDto);
  // }

  // @Delete(':id')
  // remove(@Param('id') id: string) {
  //   return this.summarizeService.remove(+id);
  // }
}
