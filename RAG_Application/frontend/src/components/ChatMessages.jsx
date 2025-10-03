import Markdown from 'react-markdown';
import useAutoScroll from '@/hooks/useAutoScroll';
import Spinner from '@/components/Spinner';
import userIcon from '@/assets/images/user.svg';
import errorIcon from '@/assets/images/error.svg';

function ChatMessages({ messages, isLoading }) {
  const scrollContentRef = useAutoScroll(isLoading);

  return (
    <div ref={scrollContentRef} className='grow space-y-4'>
      {messages.map(({ role, content, loading, error, sources }, idx) => (
        <div
          key={idx}
          className={`flex items-start gap-4 py-4 px-3 rounded-xl ${role === 'user' ? 'bg-primary-blue/10' : ''}`}
        >
          {role === 'user' && (
            <img
              className='h-[26px] w-[26px] shrink-0'
              src={userIcon}
              alt='user'
            />
          )}
          <div>
            <div className='markdown-container'>
              {(loading && !content) ? <Spinner />
                : (role === 'assistant')
                  ? <Markdown>{content}</Markdown>
                  : <div className='whitespace-pre-line'>{content}</div>
              }
            </div>

            {role === 'assistant' && Array.isArray(sources) && sources.length > 0 && (
              <div className='mt-2 text-sm opacity-80'>
                <div className='font-semibold mb-1'>Sources</div>
                <ul className='list-disc pl-5 space-y-1'>
                  {sources.map((s, i) => (
                    <li key={i}>
                      <a
                        href={import.meta.env.VITE_API_URL + s.source}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-blue-600 underline"
                      >
                        {s.source.split('/').pop()}
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {error && (
              <div className={`flex items-center gap-1 text-sm text-error-red ${content && 'mt-2'}`}>
                <img className='h-5 w-5' src={errorIcon} alt='error' />
                <span>Error generating the response</span>
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

export default ChatMessages;
