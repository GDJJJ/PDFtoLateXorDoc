from flask import Flask, render_template, request, send_file
import os
import shutil
import time
from werkzeug.utils import secure_filename
from core.document_processor import process_all_images

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './temp_uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

# 禁用 Flask 的缓存
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.after_request
def add_no_cache_headers(response):
    """为所有响应添加禁用缓存的头"""
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    response.headers['Last-Modified'] = time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.gmtime())
    return response

@app.route('/', methods=['GET'])
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        return f"<pre>错误: {str(e)}\n\n{error_msg}</pre>", 500

@app.route('/process', methods=['POST'])
def process_files():
    # 清空临时目录
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        shutil.rmtree(app.config['UPLOAD_FOLDER'])
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # 获取参数
    output_name = request.form.get('output_name', 'output')
    output_format = request.form.get('output_format', 'html')  # 默认为 html
    
    # 保存上传的文件
    uploaded_files = request.files.getlist('images')
    for file in uploaded_files:
        if file.filename != '':
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    # 处理文件
    format_extensions = {
        'html': '.html',
        'docx': '.docx',
        'pdf': '.pdf',
        'tex': '.tex'
    }
    extension = format_extensions.get(output_format, '.html')
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{output_name}{extension}')
    
    result_path = process_all_images(app.config['UPLOAD_FOLDER'], output_path, output_format)
    
    # 如果 Word 生成失败，result_path 可能是 LaTeX 文件路径
    # 处理 Path 对象转换为字符串
    if result_path:
        if hasattr(result_path, '__str__'):
            final_path = str(result_path)
        else:
            final_path = result_path
    else:
        # 回退到预期的输出文件
        final_path = output_path
    
    # 检查文件是否存在
    if not os.path.exists(final_path):
        # 尝试其他格式的文件
        for fmt in ['html', 'docx', 'tex']:
            alt_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{output_name}.{fmt}')
            if os.path.exists(alt_path):
                final_path = alt_path
                break
        else:
            return f"错误：无法生成文件。请检查服务器日志。", 500
    
    # 确定 MIME 类型
    if final_path.endswith('.docx'):
        mimetype = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    elif final_path.endswith('.pdf'):
        mimetype = 'application/pdf'
    elif final_path.endswith('.html'):
        mimetype = 'text/html'
    else:
        mimetype = 'text/plain'
    
    return send_file(final_path, as_attachment=True, mimetype=mimetype)

if __name__ == '__main__':
    app.run(debug=True)

