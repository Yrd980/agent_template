import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import apiClient from '../services/api';
import type { 
  SessionRequest, 
  MessageRequest
} from '../types/api';

// Sessions
export function useSessions() {
  return useQuery({
    queryKey: ['sessions'],
    queryFn: () => apiClient.getSessions(),
  });
}

export function useSession(id: string) {
  return useQuery({
    queryKey: ['session', id],
    queryFn: () => apiClient.getSession(id),
    enabled: !!id,
  });
}

export function useSessionMessages(id: string) {
  return useQuery({
    queryKey: ['session', id, 'messages'],
    queryFn: () => apiClient.getSessionMessages(id),
    enabled: !!id,
  });
}

export function useCreateSession() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (request: SessionRequest) => apiClient.createSession(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['sessions'] });
    },
  });
}

export function useDeleteSession() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (id: string) => apiClient.deleteSession(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['sessions'] });
    },
  });
}

// Messages
export function useSendMessage() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (request: MessageRequest) => apiClient.sendMessage(request),
    onSuccess: (_, variables) => {
      if (variables.session_id) {
        queryClient.invalidateQueries({ 
          queryKey: ['session', variables.session_id, 'messages'] 
        });
      }
    },
  });
}

// Tasks
export function useTasks(status?: string) {
  return useQuery({
    queryKey: ['tasks', status],
    queryFn: () => apiClient.getTasks(status),
  });
}

export function useTask(id: string) {
  return useQuery({
    queryKey: ['task', id],
    queryFn: () => apiClient.getTask(id),
    enabled: !!id,
  });
}

// System
export function useSystemStats() {
  return useQuery({
    queryKey: ['stats'],
    queryFn: () => apiClient.getStats(),
    refetchInterval: 5000, // Refresh every 5 seconds
  });
}

export function useHealth() {
  return useQuery({
    queryKey: ['health'],
    queryFn: () => apiClient.getHealth(),
    refetchInterval: 30000, // Refresh every 30 seconds
  });
}

// Tools
export function useTools() {
  return useQuery({
    queryKey: ['tools'],
    queryFn: () => apiClient.getTools(),
  });
}