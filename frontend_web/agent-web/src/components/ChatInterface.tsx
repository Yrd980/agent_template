import React, { useState, useRef, useEffect } from 'react';
import { Send, Plus, Loader2 } from 'lucide-react';
import { useSessionMessages, useSendMessage, useCreateSession } from '../hooks/useApi';
import { useWebSocketEvent } from '../hooks/useWebSocket';
import type { Message, MessageRole } from '../types/api';

interface ChatInterfaceProps {
  sessionId: string | null;
  onSessionChange: (sessionId: string | null) => void;
}

export default function ChatInterface({ sessionId, onSessionChange }: ChatInterfaceProps) {
  const [message, setMessage] = useState('');
  const [streamingContent, setStreamingContent] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  const { data: messages = [], refetch } = useSessionMessages(sessionId || '');
  const sendMessageMutation = useSendMessage();
  const createSessionMutation = useCreateSession();

  // WebSocket event handlers
  useWebSocketEvent('message', (newMessage: Message) => {
    if (newMessage.session_id === sessionId) {
      refetch();
    }
  });

  useWebSocketEvent('stream', (streamData) => {
    if (streamData.session_id === sessionId) {
      if (streamData.is_complete) {
        setIsStreaming(false);
        setStreamingContent('');
        refetch();
      } else {
        setStreamingContent(prev => prev + streamData.content);
      }
    }
  });

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamingContent]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!message.trim()) return;

    let currentSessionId = sessionId;

    // Create session if none exists
    if (!currentSessionId) {
      try {
        const newSession = await createSessionMutation.mutateAsync({});
        currentSessionId = newSession.id;
        onSessionChange(currentSessionId);
      } catch (error) {
        console.error('Failed to create session:', error);
        return;
      }
    }

    const messageContent = message;
    setMessage('');
    setIsStreaming(true);

    try {
      await sendMessageMutation.mutateAsync({
        content: messageContent,
        session_id: currentSessionId,
        role: 'user' as MessageRole,
      });
    } catch (error) {
      console.error('Failed to send message:', error);
      setIsStreaming(false);
    }
  };

  const handleNewSession = async () => {
    try {
      const newSession = await createSessionMutation.mutateAsync({});
      onSessionChange(newSession.id);
    } catch (error) {
      console.error('Failed to create session:', error);
    }
  };

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString();
  };

  const getRoleColor = (role: MessageRole) => {
    switch (role) {
      case 'user':
        return 'bg-blue-500';
      case 'assistant':
        return 'bg-green-500';
      case 'system':
        return 'bg-gray-500';
      case 'tool':
        return 'bg-purple-500';
      default:
        return 'bg-gray-500';
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Chat Header */}
      <div className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <h2 className="text-lg font-semibold text-gray-900">
              {sessionId ? `Session: ${sessionId.slice(0, 8)}...` : 'New Chat'}
            </h2>
            {messages.length > 0 && (
              <span className="text-sm text-gray-500">
                {messages.length} messages
              </span>
            )}
          </div>
          <button
            onClick={handleNewSession}
            disabled={createSessionMutation.isPending}
            className="flex items-center space-x-2 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50"
          >
            <Plus className="w-4 h-4" />
            <span>New Session</span>
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-6 space-y-4">
        {messages.length === 0 && !isStreaming && (
          <div className="text-center text-gray-500 mt-20">
            <div className="text-6xl mb-4">🤖</div>
            <h3 className="text-xl font-medium mb-2">Start a conversation</h3>
            <p>Send a message to begin chatting with the AI agent.</p>
          </div>
        )}

        {messages.map((msg) => (
          <div key={msg.id} className="flex items-start space-x-4">
            <div className={`w-8 h-8 rounded-full ${getRoleColor(msg.role)} flex items-center justify-center text-white text-sm font-medium`}>
              {msg.role.charAt(0).toUpperCase()}
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center space-x-2 mb-1">
                <span className="font-medium text-gray-900 capitalize">
                  {msg.role}
                </span>
                <span className="text-xs text-gray-500">
                  {formatTimestamp(msg.timestamp)}
                </span>
              </div>
              <div className="prose prose-sm max-w-none">
                <pre className="whitespace-pre-wrap text-gray-700 font-sans">
                  {msg.content}
                </pre>
              </div>
            </div>
          </div>
        ))}

        {/* Streaming message */}
        {isStreaming && streamingContent && (
          <div className="flex items-start space-x-4">
            <div className="w-8 h-8 rounded-full bg-green-500 flex items-center justify-center text-white text-sm font-medium">
              A
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center space-x-2 mb-1">
                <span className="font-medium text-gray-900">Assistant</span>
                <Loader2 className="w-3 h-3 animate-spin text-green-500" />
              </div>
              <div className="prose prose-sm max-w-none">
                <pre className="whitespace-pre-wrap text-gray-700 font-sans">
                  {streamingContent}
                </pre>
              </div>
            </div>
          </div>
        )}

        {isStreaming && !streamingContent && (
          <div className="flex items-center space-x-2 text-gray-500">
            <Loader2 className="w-4 h-4 animate-spin" />
            <span>AI is thinking...</span>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Message Input */}
      <div className="bg-white border-t border-gray-200 p-6">
        <form onSubmit={handleSubmit} className="flex items-end space-x-4">
          <div className="flex-1">
            <textarea
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              placeholder="Type your message here..."
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-none"
              rows={3}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSubmit(e);
                }
              }}
            />
          </div>
          <button
            type="submit"
            disabled={!message.trim() || isStreaming || sendMessageMutation.isPending}
            className="px-6 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
          >
            {sendMessageMutation.isPending ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Send className="w-4 h-4" />
            )}
            <span>Send</span>
          </button>
        </form>
      </div>
    </div>
  );
}