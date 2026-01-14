import { Controller, Get, Post, Body, Param, Sse } from '@nestjs/common';
import { ChatService } from './chat.service';
import { CreateChatDto } from './dto/create-chat.dto';
import { Observable } from 'rxjs';

@Controller('chat')
export class ChatController {
  constructor(private readonly chatService: ChatService) {}

  @Post()
  create(@Body() createChatDto: CreateChatDto) {
    return this.chatService.create(createChatDto);
  }

  @Post('stream')
  @Sse()
  async stream(@Body() createChatDto: CreateChatDto): Promise<Observable<MessageEvent>> {
    return this.chatService.createStream(createChatDto);
  }

  @Get('/history/:summaryId')
  history(@Param('summaryId') summaryId: string) {
    return this.chatService.history(summaryId);
  }
}
