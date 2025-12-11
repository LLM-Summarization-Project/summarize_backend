import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';

export interface CreateTopicDto {
    userId: number;
    name: string;
    description?: string;
}

export interface OntologyTopicResponse {
    success: boolean;
    data?: any;
    error?: string;
}

@Injectable()
export class OntologyService {
    private readonly logger = new Logger(OntologyService.name);
    private readonly ontologyBaseUrl: string;

    constructor(private configService: ConfigService) {
        this.ontologyBaseUrl =
            this.configService.get<string>('ONTOLOGY_SERVICE_URL') ??
            'http://localhost:3000';
    }

    /**
     * Trigger ontology service to create a topic
     * POST /ontology/topic
     */
    async createTopic(dto: CreateTopicDto): Promise<OntologyTopicResponse> {
        const url = `${this.ontologyBaseUrl}/ontology/topic`;

        try {
            this.logger.log(`Calling ontology service: POST ${url}`);

            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(dto),
            });

            if (!response.ok) {
                const errorText = await response.text();
                this.logger.error(
                    `Ontology service error: ${response.status} - ${errorText}`,
                );
                return {
                    success: false,
                    error: `Ontology service returned ${response.status}: ${errorText}`,
                };
            }

            const data = await response.json();
            this.logger.log(`Ontology service response: ${JSON.stringify(data)}`);

            return {
                success: true,
                data,
            };
        } catch (error) {
            this.logger.error(`Failed to call ontology service: ${error}`);
            return {
                success: false,
                error: error instanceof Error ? error.message : 'Unknown error',
            };
        }
    }

    /**
     * Trigger ontology topic creation and return combined result
     * Use this before sending response to frontend
     */
    async triggerAndGetResult<T>(
        topicData: CreateTopicDto,
        originalData: T,
    ): Promise<{ originalData: T; ontologyResult: OntologyTopicResponse }> {
        const ontologyResult = await this.createTopic(topicData);
        return {
            originalData,
            ontologyResult,
        };
    }
}
