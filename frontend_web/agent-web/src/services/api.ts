import type {
  Message,
  MessageRequest,
  MessageResponse,
  Session,
  SessionRequest,
  Task,
  TaskRequest,
  ToolInfo,
  ToolCallRequest,
  SystemStats,
} from '../types/api';

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = 'http://127.0.0.1:8000') {
    this.baseUrl = baseUrl;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return response.json();
  }

  // Health check
  async getHealth(): Promise<{ status: string; timestamp: string }> {
    return this.request('/health');
  }

  // System stats
  async getStats(): Promise<SystemStats> {
    return this.request('/api/v1/stats');
  }

  // Sessions
  async createSession(request: SessionRequest): Promise<Session> {
    return this.request('/api/v1/sessions', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async getSessions(): Promise<Session[]> {
    return this.request('/api/v1/sessions');
  }

  async getSession(id: string): Promise<Session> {
    return this.request(`/api/v1/sessions/${id}`);
  }

  async deleteSession(id: string): Promise<void> {
    return this.request(`/api/v1/sessions/${id}`, {
      method: 'DELETE',
    });
  }

  async getSessionMessages(id: string): Promise<Message[]> {
    return this.request(`/api/v1/sessions/${id}/messages`);
  }

  // Messages
  async sendMessage(request: MessageRequest): Promise<MessageResponse> {
    return this.request('/api/v1/messages', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async streamMessage(request: MessageRequest): Promise<ReadableStream<Uint8Array>> {
    const response = await fetch(`${this.baseUrl}/api/v1/messages/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return response.body!;
  }

  // Tasks
  async createTask(request: TaskRequest): Promise<Task> {
    return this.request('/api/v1/tasks', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async getTasks(status?: string): Promise<Task[]> {
    const params = status ? `?status=${status}` : '';
    return this.request(`/api/v1/tasks${params}`);
  }

  async getTask(id: string): Promise<Task> {
    return this.request(`/api/v1/tasks/${id}`);
  }

  async cancelTask(id: string): Promise<void> {
    return this.request(`/api/v1/tasks/${id}`, {
      method: 'DELETE',
    });
  }

  // Tools
  async getTools(): Promise<ToolInfo[]> {
    return this.request('/api/v1/tools');
  }

  async callTool(request: ToolCallRequest): Promise<any> {
    return this.request('/api/v1/tools/call', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }
}

export const apiClient = new ApiClient();
export default apiClient;