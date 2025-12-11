import { Global, Module } from '@nestjs/common';
import { OntologyService } from './ontology.service';

@Global()
@Module({
    providers: [OntologyService],
    exports: [OntologyService],
})
export class OntologyModule { }
