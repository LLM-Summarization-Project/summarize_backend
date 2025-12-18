import { Injectable, CanActivate, ExecutionContext, UnauthorizedException } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { JwtService } from '@nestjs/jwt';
import { Request } from 'express';

@Injectable()
export class LocalJwtAuthGuard implements CanActivate {
  constructor(private readonly jwtService: JwtService, private readonly configService: ConfigService) {}

  private extractTokenFromRequest(req: Request): string | null {
    if (req && req.cookies && req.cookies['access_token']) {
      return req.cookies['access_token'];
    }

    const auth = req.headers['authorization'] as string;
    if (auth && auth.startsWith('Bearer ')) return auth.split(' ')[1];

    return null;
  }

  canActivate(context: ExecutionContext) {
    const req = context.switchToHttp().getRequest<Request>();
    const token = this.extractTokenFromRequest(req);

    if (!token) throw new UnauthorizedException('No token provided');

    try {
      const payload = this.jwtService.verify(token, {
        secret: this.configService.get<string>('JWT_SECRET'),
      });

      (req as any).user = { id: payload.sub, username: payload.username, ...payload };
      return true;
    } catch (err) {
      throw new UnauthorizedException('Invalid or expired token');
    }
  }
}
