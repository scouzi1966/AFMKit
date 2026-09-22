// Frozen native templates for existing Qwen families sharing Qwen3VLProcessor.
// Regression fixtures only: no model weights or runtime downloads required.
enum QwenSharedProcessorTemplateFixtures {
    // Source: mlx-community/Qwen3.5-35B-A3B-4bit/chat_template.jinja
    // Source SHA256: a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715
    static let moe = #"""
    {%- set image_count = namespace(value=0) %}
    {%- set video_count = namespace(value=0) %}
    {%- macro render_content(content, do_vision_count, is_system_content=false) %}
        {%- if content is string %}
            {{- content }}
        {%- elif content is iterable and content is not mapping %}
            {%- for item in content %}
                {%- if 'image' in item or 'image_url' in item or item.type == 'image' %}
                    {%- if is_system_content %}
                        {{- raise_exception('System message cannot contain images.') }}
                    {%- endif %}
                    {%- if do_vision_count %}
                        {%- set image_count.value = image_count.value + 1 %}
                    {%- endif %}
                    {%- if add_vision_id %}
                        {{- 'Picture ' ~ image_count.value ~ ': ' }}
                    {%- endif %}
                    {{- '<|vision_start|><|image_pad|><|vision_end|>' }}
                {%- elif 'video' in item or item.type == 'video' %}
                    {%- if is_system_content %}
                        {{- raise_exception('System message cannot contain videos.') }}
                    {%- endif %}
                    {%- if do_vision_count %}
                        {%- set video_count.value = video_count.value + 1 %}
                    {%- endif %}
                    {%- if add_vision_id %}
                        {{- 'Video ' ~ video_count.value ~ ': ' }}
                    {%- endif %}
                    {{- '<|vision_start|><|video_pad|><|vision_end|>' }}
                {%- elif 'text' in item %}
                    {{- item.text }}
                {%- else %}
                    {{- raise_exception('Unexpected item type in content.') }}
                {%- endif %}
            {%- endfor %}
        {%- elif content is none or content is undefined %}
            {{- '' }}
        {%- else %}
            {{- raise_exception('Unexpected content type.') }}
        {%- endif %}
    {%- endmacro %}
    {%- if not messages %}
        {{- raise_exception('No messages provided.') }}
    {%- endif %}
    {%- if tools and tools is iterable and tools is not mapping %}
        {{- '<|im_start|>system\n' }}
        {{- "# Tools\n\nYou have access to the following functions:\n\n<tools>" }}
        {%- for tool in tools %}
            {{- "\n" }}
            {{- tool | tojson }}
        {%- endfor %}
        {{- "\n</tools>" }}
        {{- '\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>' }}
        {%- if messages[0].role == 'system' %}
            {%- set content = render_content(messages[0].content, false, true)|trim %}
            {%- if content %}
                {{- '\n\n' + content }}
            {%- endif %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- else %}
        {%- if messages[0].role == 'system' %}
            {%- set content = render_content(messages[0].content, false, true)|trim %}
            {{- '<|im_start|>system\n' + content + '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
    {%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
    {%- for message in messages[::-1] %}
        {%- set index = (messages|length - 1) - loop.index0 %}
        {%- if ns.multi_step_tool and message.role == "user" %}
            {%- set content = render_content(message.content, false)|trim %}
            {%- if not(content.startswith('<tool_response>') and content.endswith('</tool_response>')) %}
                {%- set ns.multi_step_tool = false %}
                {%- set ns.last_query_index = index %}
            {%- endif %}
        {%- endif %}
    {%- endfor %}
    {%- if ns.multi_step_tool %}
        {{- raise_exception('No user query found in messages.') }}
    {%- endif %}
    {%- for message in messages %}
        {%- set content = render_content(message.content, true)|trim %}
        {%- if message.role == "system" %}
            {%- if not loop.first %}
                {{- raise_exception('System message must be at the beginning.') }}
            {%- endif %}
        {%- elif message.role == "user" %}
            {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
        {%- elif message.role == "assistant" %}
            {%- set reasoning_content = '' %}
            {%- if message.reasoning_content is string %}
                {%- set reasoning_content = message.reasoning_content %}
            {%- else %}
                {%- if '</think>' in content %}
                    {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
                    {%- set content = content.split('</think>')[-1].lstrip('\n') %}
                {%- endif %}
            {%- endif %}
            {%- set reasoning_content = reasoning_content|trim %}
            {%- if loop.index0 > ns.last_query_index %}
                {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content + '\n</think>\n\n' + content }}
            {%- else %}
                {{- '<|im_start|>' + message.role + '\n' + content }}
            {%- endif %}
            {%- if message.tool_calls and message.tool_calls is iterable and message.tool_calls is not mapping %}
                {%- for tool_call in message.tool_calls %}
                    {%- if tool_call.function is defined %}
                        {%- set tool_call = tool_call.function %}
                    {%- endif %}
                    {%- if loop.first %}
                        {%- if content|trim %}
                            {{- '\n\n<tool_call>\n<function=' + tool_call.name + '>\n' }}
                        {%- else %}
                            {{- '<tool_call>\n<function=' + tool_call.name + '>\n' }}
                        {%- endif %}
                    {%- else %}
                        {{- '\n<tool_call>\n<function=' + tool_call.name + '>\n' }}
                    {%- endif %}
                    {%- if tool_call.arguments is defined %}
                        {%- for args_name, args_value in tool_call.arguments|items %}
                            {{- '<parameter=' + args_name + '>\n' }}
                            {%- set args_value = args_value | tojson | safe if args_value is mapping or (args_value is sequence and args_value is not string) else args_value | string %}
                            {{- args_value }}
                            {{- '\n</parameter>\n' }}
                        {%- endfor %}
                    {%- endif %}
                    {{- '</function>\n</tool_call>' }}
                {%- endfor %}
            {%- endif %}
            {{- '<|im_end|>\n' }}
        {%- elif message.role == "tool" %}
            {%- if loop.previtem and loop.previtem.role != "tool" %}
                {{- '<|im_start|>user' }}
            {%- endif %}
            {{- '\n<tool_response>\n' }}
            {{- content }}
            {{- '\n</tool_response>' }}
            {%- if not loop.last and loop.nextitem.role != "tool" %}
                {{- '<|im_end|>\n' }}
            {%- elif loop.last %}
                {{- '<|im_end|>\n' }}
            {%- endif %}
        {%- else %}
            {{- raise_exception('Unexpected message role.') }}
        {%- endif %}
    {%- endfor %}
    {%- if add_generation_prompt %}
        {{- '<|im_start|>assistant\n' }}
        {%- if enable_thinking is defined and enable_thinking is false %}
            {{- '<think>\n\n</think>\n\n' }}
        {%- else %}
            {{- '<think>\n' }}
        {%- endif %}
    {%- endif %}
    """#

    // Source: mlx-community/Qwen3-VL-4B-Instruct-4bit@2fd8dacbdb8f1e54b8c005f081ec5bf79c56376b/chat_template.jinja
    // Source SHA256: 3636d0f0bd6bef02654cdffdc447b79cb2cef8ab02cc75267345946291a489e4
    static let vl = #"""
    {%- if tools %}
        {{- '<|im_start|>system\n' }}
        {%- if messages[0].role == 'system' %}
            {%- if messages[0].content is string %}
                {{- messages[0].content }}
            {%- else %}
                {%- for content in messages[0].content %}
                    {%- if 'text' in content %}
                        {{- content.text }}
                    {%- endif %}
                {%- endfor %}
            {%- endif %}
            {{- '\n\n' }}
        {%- endif %}
        {{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
        {%- for tool in tools %}
            {{- "\n" }}
            {{- tool | tojson }}
        {%- endfor %}
        {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
    {%- else %}
        {%- if messages[0].role == 'system' %}
            {{- '<|im_start|>system\n' }}
            {%- if messages[0].content is string %}
                {{- messages[0].content }}
            {%- else %}
                {%- for content in messages[0].content %}
                    {%- if 'text' in content %}
                        {{- content.text }}
                    {%- endif %}
                {%- endfor %}
            {%- endif %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
    {%- set image_count = namespace(value=0) %}
    {%- set video_count = namespace(value=0) %}
    {%- for message in messages %}
        {%- if message.role == "user" %}
            {{- '<|im_start|>' + message.role + '\n' }}
            {%- if message.content is string %}
                {{- message.content }}
            {%- else %}
                {%- for content in message.content %}
                    {%- if content.type == 'image' or 'image' in content or 'image_url' in content %}
                        {%- set image_count.value = image_count.value + 1 %}
                        {%- if add_vision_id %}Picture {{ image_count.value }}: {% endif -%}
                        <|vision_start|><|image_pad|><|vision_end|>
                    {%- elif content.type == 'video' or 'video' in content %}
                        {%- set video_count.value = video_count.value + 1 %}
                        {%- if add_vision_id %}Video {{ video_count.value }}: {% endif -%}
                        <|vision_start|><|video_pad|><|vision_end|>
                    {%- elif 'text' in content %}
                        {{- content.text }}
                    {%- endif %}
                {%- endfor %}
            {%- endif %}
            {{- '<|im_end|>\n' }}
        {%- elif message.role == "assistant" %}
            {{- '<|im_start|>' + message.role + '\n' }}
            {%- if message.content is string %}
                {{- message.content }}
            {%- else %}
                {%- for content_item in message.content %}
                    {%- if 'text' in content_item %}
                        {{- content_item.text }}
                    {%- endif %}
                {%- endfor %}
            {%- endif %}
            {%- if message.tool_calls %}
                {%- for tool_call in message.tool_calls %}
                    {%- if (loop.first and message.content) or (not loop.first) %}
                        {{- '\n' }}
                    {%- endif %}
                    {%- if tool_call.function %}
                        {%- set tool_call = tool_call.function %}
                    {%- endif %}
                    {{- '<tool_call>\n{"name": "' }}
                    {{- tool_call.name }}
                    {{- '", "arguments": ' }}
                    {%- if tool_call.arguments is string %}
                        {{- tool_call.arguments }}
                    {%- else %}
                        {{- tool_call.arguments | tojson }}
                    {%- endif %}
                    {{- '}\n</tool_call>' }}
                {%- endfor %}
            {%- endif %}
            {{- '<|im_end|>\n' }}
        {%- elif message.role == "tool" %}
            {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
                {{- '<|im_start|>user' }}
            {%- endif %}
            {{- '\n<tool_response>\n' }}
            {%- if message.content is string %}
                {{- message.content }}
            {%- else %}
                {%- for content in message.content %}
                    {%- if content.type == 'image' or 'image' in content or 'image_url' in content %}
                        {%- set image_count.value = image_count.value + 1 %}
                        {%- if add_vision_id %}Picture {{ image_count.value }}: {% endif -%}
                        <|vision_start|><|image_pad|><|vision_end|>
                    {%- elif content.type == 'video' or 'video' in content %}
                        {%- set video_count.value = video_count.value + 1 %}
                        {%- if add_vision_id %}Video {{ video_count.value }}: {% endif -%}
                        <|vision_start|><|video_pad|><|vision_end|>
                    {%- elif 'text' in content %}
                        {{- content.text }}
                    {%- endif %}
                {%- endfor %}
            {%- endif %}
            {{- '\n</tool_response>' }}
            {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
                {{- '<|im_end|>\n' }}
            {%- endif %}
        {%- endif %}
    {%- endfor %}
    {%- if add_generation_prompt %}
        {{- '<|im_start|>assistant\n' }}
    {%- endif %}
    
    """#

}

