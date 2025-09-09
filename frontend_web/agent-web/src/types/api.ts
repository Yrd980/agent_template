// TypeScript interfaces matching Python Pydantic models

export interface Message {
  id: string;
  type: MessageType;
  content: string;
  session_id: string;
  role: MessageRole;
  timestamp: string;
  metadata: Record<string, any>;
}

export interface MessageRequest {
  content: string;
  session_id?: string;
  role?: MessageRole;
  message_type?: MessageType;
  metadata?: Record<string, any>;
}

export interface MessageResponse {
  id: string;
  type: MessageType;
  content: string;
  session_id: string;
  timestamp: string;
  metadata: Record<string, any>;
}

export interface Session {
  id: string;
  created_at: string;
  updated_at: string;
  message_count: number;
  metadata: Record<string, any>;
}

export interface SessionRequest {
  metadata?: Record<string, any>;
}

export interface Task {
  id: string;
  type: TaskType;
  status: TaskStatus;
  priority: number;
  description: string;
  created_at: string;
  updated_at: string;
  metadata: Record<string, any>;
  result?: any;
  error?: string;
}

export interface TaskRequest {
  type: TaskType;
  priority?: number;
  description: string;
  metadata?: Record<string, any>;
}

export interface ToolInfo {
  name: string;
  description: string;
  parameters: Record<string, any>;
  metadata: Record<string, any>;
}

export interface ToolCallRequest {
  name: string;
  parameters: Record<string, any>;
  session_id?: string;
}

export interface SystemStats {
  uptime: number;
  active_sessions: number;
  total_messages: number;
  active_tasks: number;
  completed_tasks: number;
  failed_tasks: number;
  memory_usage: number;
  cpu_usage: number;
  websocket_connections: number;
}

// Type unions instead of enums for compatibility
export type MessageType = 'text' | 'tool_call' | 'tool_result' | 'system' | 'error';
export type MessageRole = 'user' | 'assistant' | 'system' | 'tool';
export type TaskType = 'message_processing' | 'tool_execution' | 'subagent_task' | 'system_task';
export type TaskStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';

// WebSocket Events
export interface WebSocketEvent {
  type: string;
  data: any;
  timestamp: string;
}

export interface MessageEvent extends WebSocketEvent {
  type: 'message';
  data: Message;
}

export interface TaskEvent extends WebSocketEvent {
  type: 'task_update';
  data: Task;
}

export interface StreamEvent extends WebSocketEvent {
  type: 'stream';
  data: {
    session_id: string;
    content: string;
    is_complete: boolean;
  };
}

export interface SessionEvent extends WebSocketEvent {
  type: 'session_update';
  data: Session;
}