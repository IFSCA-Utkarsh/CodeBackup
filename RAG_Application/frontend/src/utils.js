import { EventSourceParserStream } from 'eventsource-parser/stream';

export async function* parseSSEStream(stream) {
  const reader = stream
    .pipeThrough(new TextDecoderStream())
    .pipeThrough(new EventSourceParserStream())
    .getReader();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (!value) continue;

    const data = value.data;
    if (data === '[DONE]') {
      yield { done: true };
      break;
    }

    try {
      yield { done: false, payload: JSON.parse(data) };
    } catch {
      // Tolerate non-JSON messages
      yield { done: false, payload: { text: data } };
    }
  }
}
