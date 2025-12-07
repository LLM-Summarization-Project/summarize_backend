import { Module } from '@nestjs/common';
import { JwtModule } from '@nestjs/jwt';
import { LocalJwtAuthGuard } from './jwt-auth.guard';

@Module({
  imports: [
    JwtModule.register({
      secret: process.env.JWT_SECRET,
      signOptions: { expiresIn: '15m' },
    }),
  ],  
  providers: [LocalJwtAuthGuard],
  exports: [LocalJwtAuthGuard, JwtModule],
})
export class AuthModule {}
