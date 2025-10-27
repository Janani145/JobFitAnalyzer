# app.py
import os, re, uuid, zipfile, io, docx2txt, PyPDF2, json
from datetime import datetime
from flask import Flask, request, render_template, send_from_directory, redirect, url_for, session, flash, send_file
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import pandas as pd
from wordcloud import WordCloud
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas as pdf_canvas
from PIL import Image
import numpy as np


# Optional: if you want better semantic matching, install sentence-transformers and uncomment below
# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('all-MiniLM-L6-v2')

# ---- CONFIG ----
app = Flask(__name__)
app.secret_key = os.environ.get('APP_SECRET', 'supersecretkey')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['REPORT_FOLDER'] = 'reports'
app.config['STATIC_FOLDER'] = 'static'
app.config['ALLOWED_EXT'] = {'.pdf', '.docx', '.txt'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['REPORT_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['STATIC_FOLDER'],'images'), exist_ok=True)

ADMIN_PASS = os.environ.get('ADMIN_PASS', 'admin123')  # change in production

# ---- SKILL LIBRARY (extendable) ----
SKILLS_DB = [
    "python","java","c++","c#","html","css","javascript","react","angular","vue",
    "nodejs","node.js","express","flask","django","sql","mysql","postgresql","mongodb",
    "nosql","machine learning","deep learning","data analysis","nlp","natural language processing",
    "tensorflow","pytorch","scikit-learn","pandas","numpy","git","github","docker","kubernetes",
    "aws","azure","gcp","rest api","api development","microservices","communication","leadership",
    "problem solving","teamwork","project management","excel","tableau","power bi","spark","hadoop",
    "selenium","pytest"
]

# synonyms mapping: common abbreviations
SYNONYMS = {
    "ml": "machine learning",
    "dl": "deep learning",
    "js": "javascript",
    "node": "nodejs",
    "tf": "tensorflow",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "ai": "machine learning",
    "pm": "project management"
}

# simple role mapping for career suggestions
ROLE_MAP = {
    "data analyst": ["python","sql","pandas","excel","tableau"],
    "data scientist": ["python","machine learning","pandas","numpy","scikit-learn","tensorflow"],
    "backend developer": ["python","flask","django","nodejs","sql","rest api"],
    "frontend developer": ["html","css","javascript","react","angular","vue"],
    "devops engineer": ["docker","kubernetes","aws","gcp"],
    "qa engineer": ["selenium","pytest"]
}

TRAINING_MAP = {
    "python": ["https://docs.python.org/3/tutorial/", "https://www.learnpython.org/"],
    "sql": ["https://mode.com/sql-tutorial/","https://www.w3schools.com/sql/"],
    "machine learning": ["https://www.coursera.org/learn/machine-learning","https://www.fast.ai/"],
    "docker": ["https://www.docker.com/get-started"],
    "kubernetes": ["https://kubernetes.io/docs/tutorials/"]
}

# ---- HELPERS: TEXT EXTRACTION ----
def extract_text(file_path):
    lower = file_path.lower()
    if lower.endswith('.pdf'):
        text = ""
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                try:
                    page_text = page.extract_text()
                except Exception:
                    page_text = ""
                if page_text: text += page_text + "\n"
        return text
    elif lower.endswith('.docx'):
        try:
            return docx2txt.process(file_path) or ""
        except Exception:
            return ""
    elif lower.endswith('.txt'):
        with open(file_path,'r',encoding='utf-8',errors='ignore') as f:
            return f.read()
    return ""

# ---- ANONYMIZATION ----
EMAIL_RE = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
PHONE_RE = re.compile(r'(\+?\d{1,3}[\s\-]?)?(\(?\d{2,4}\)?[\s\-]?)?[\d\s\-]{6,15}')
NAME_LINE_RE = re.compile(r'^(name[:\-\s].+)$', flags=re.IGNORECASE|re.MULTILINE)
LINKEDIN_RE = re.compile(r'(https?://)?(www\.)?linkedin\.com/[\w/.-]+', flags=re.IGNORECASE)
ADDRESS_RE = re.compile(r'\d{1,4}\s+\w+\s+(street|st|rd|road|avenue|ave|lane|ln)\b', flags=re.IGNORECASE)
DATE_RE = re.compile(r'\b(?:\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}|\bJan(?:uary)?|\bFeb(?:ruary)?|\bMar(?:ch)?|\bApr(?:il)?|\bMay\b|\bJun(?:e)?|\bJul(?:y)?|\bAug(?:ust)?|\bSep(?:tember)?|\bOct(?:ober)?|\bNov(?:ember)?|\bDec(?:ember)?)', flags=re.IGNORECASE)

def anonymize_text(text, candidate_id=None):
    text = EMAIL_RE.sub('[email_removed]', text)
    text = PHONE_RE.sub('[phone_removed]', text)
    text = NAME_LINE_RE.sub('[name_removed]', text)
    text = LINKEDIN_RE.sub('[link_removed]', text)
    text = ADDRESS_RE.sub('[address_removed]', text)
    text = DATE_RE.sub('[date_removed]', text)
    header = f"Candidate ID: {candidate_id}\n"
    return header + text

# ---- EXPERIENCE & EDUCATION EXTRACTION (heuristic) ----
def extract_experience_and_education(text):
    years = []
    exp_years = None
    # find patterns like 2015-2019, 2018 - present
    ranges = re.findall(r'(\b\d{4}\b)\s*[-–]\s*(\b\d{4}\b|present|current)', text, flags=re.IGNORECASE)
    for start, end in ranges:
        try:
            s = int(start)
            if end.lower().isdigit():
                e = int(end)
                years.append(max(0, e - s))
        except:
            continue
    # look for "X years" patterns
    yrs = re.findall(r'(\d{1,2})\s+years?', text, flags=re.IGNORECASE)
    yrs = [int(x) for x in yrs if int(x) < 60]
    if yrs:
        exp_years = max(yrs)
    elif years:
        exp_years = max(years)
    # degree detection
    degree_keywords = ['bachelor','b.a.','b.sc','btech','b.e.','m.sc','mtech','mba','phd','master','doctor']
    degrees = []
    for kw in degree_keywords:
        if kw in text.lower():
            degrees.append(kw)
    return {'years_experience': exp_years or 0, 'degrees': list(set(degrees))}

# ---- SKILL EXTRACTION + SEMANTIC/FUZZY MATCH ----
def normalize_token(tok):
    tok = tok.strip().lower()
    tok = SYNONYMS.get(tok, tok)
    return tok

def extract_skills_from_text(text):
    text_lower = text.lower()
    found = set()
    # direct substrings
    for skill in SKILLS_DB:
        if skill in text_lower:
            found.add(skill)
    # token lookups (1-3 words)
    words = re.findall(r'[a-zA-Z0-9\.\+#]+(?: [a-zA-Z0-9\.\+#]+){0,2}', text_lower)
    for w in words:
        w_norm = normalize_token(w)
        matches = get_close_matches(w_norm, SKILLS_DB, n=1, cutoff=0.85)
        if matches:
            found.add(matches[0])
        else:
            # check synonyms mapping and substring
            if w_norm in SYNONYMS:
                found.add(SYNONYMS[w_norm])
            else:
                # partial match
                for skill in SKILLS_DB:
                    if w_norm in skill:
                        found.add(skill)
    return sorted(found)

def suggest_skills_from_jd(job_description, top_n=12):
    found = extract_skills_from_text(job_description)
    if found:
        return found
    # fallback TF-IDF keywords
    vectorizer = TfidfVectorizer(stop_words='english', max_features=50)
    vect = vectorizer.fit_transform([job_description])
    feature_names = vectorizer.get_feature_names_out()
    scores = vect.toarray()[0]
    top_indices = scores.argsort()[-top_n:][::-1]
    suggestions = [feature_names[i] for i in top_indices if scores[i]>0]
    return suggestions or ["No clear skills detected. Please refine the job description."]

# ---- SCORING ----
def compute_scores(job_description, candidate_texts, candidate_infos, critical_skills=None,
                   weight_similarity=0.6, weight_skill_coverage=0.4):
    # vector TF-IDF similarity (job vs resumes)
    corpus = [job_description] + candidate_texts
    vectorizer = TfidfVectorizer(stop_words='english').fit_transform(corpus)
    vectors = vectorizer.toarray()
    job_vector = vectors[0]
    resume_vectors = vectors[1:]
    similarities = cosine_similarity([job_vector],resume_vectors)[0]

    suggested_skills = suggest_skills_from_jd(job_description)
    req_skills = set([s.lower() for s in suggested_skills if isinstance(s,str)])

    top_candidates = []
    for i, info in enumerate(candidate_infos):
        res_skills = set([s.lower() for s in info.get('skills',[])])
        matched = sorted([s for s in res_skills if s in req_skills])
        missing = sorted([s for s in req_skills if s not in res_skills])
        # if critical skills provided, boost coverage
        coverage = (len(matched) / max(len(req_skills),1)) * 100
        if critical_skills:
            crit = set([s.lower() for s in critical_skills])
            crit_found = len([s for s in crit if s in res_skills])
            crit_total = max(len(crit),1)
            crit_score = (crit_found / crit_total)  # 0..1
        else:
            crit_score = 1.0
        skill_score = (coverage/100) * 0.8 + crit_score * 0.2  # blend coverage & critical presence
        final_score = weight_similarity * float(similarities[i]) + weight_skill_coverage * skill_score
        candidate = {
            'id': info['id'],
            'orig_filename': info['orig_filename'],
            'score': round(final_score,4),
            'sim_score': round(float(similarities[i]),4),
            'matched_skills': matched,
            'missing_skills': missing,
            'coverage': round(coverage),
            'years_experience': info.get('years_experience',0),
            'degrees': info.get('degrees',[]),
            'text_excerpt': (info.get('raw_text') or "")[:800]
        }
        top_candidates.append(candidate)
    return top_candidates, suggested_skills

# ---- PDF REPORTS ----
def make_pdf_report(report_name, job_description, top_candidates):
    filepath = os.path.join(app.config['REPORT_FOLDER'], report_name)
    c = canvas.Canvas(filepath, pagesize=A4)
    width,height = A4
    x_margin,y = 40,height-50
    c.setFont("Helvetica-Bold",16)
    c.drawString(x_margin,y,"Resume Matcher Report")
    c.setFont("Helvetica",10)
    y-=25
    c.drawString(x_margin,y,f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y-=20
    jd_text = " ".join(job_description.strip().splitlines())[:1200]
    c.setFont("Helvetica-Bold",12)
    c.drawString(x_margin,y,"Job Description (truncated)")
    y-=14
    text_obj = c.beginText(x_margin,y)
    text_obj.setFont("Helvetica",9)
    for chunk in re.findall('.{1,120}',jd_text):
        text_obj.textLine(chunk)
        y-=10
    c.drawText(text_obj)
    y-=20
    for idx,cand in enumerate(top_candidates,start=1):
        if y<140:
            c.showPage()
            y=height-50
        c.setFont("Helvetica-Bold",11)
        c.drawString(x_margin,y,f"{idx}. Candidate ID: {cand['id']} — Score: {cand['score']}")
        y-=14
        c.setFont("Helvetica",9)
        c.drawString(x_margin,y,f"Original Resume: {cand['orig_filename']}")
        y-=12
        c.drawString(x_margin,y,f"SimScore: {cand['sim_score']}, Coverage: {cand['coverage']}%")
        y-=12
        c.drawString(x_margin,y,f"Matched: {', '.join(cand['matched_skills']) or 'None'}")
        y-=12
        c.drawString(x_margin,y,f"Missing: {', '.join(cand['missing_skills']) or 'None'}")
        y-=12
        c.drawString(x_margin,y,f"Exp: {cand.get('years_experience',0)} years | Degrees: {', '.join(cand.get('degrees',[])) or 'N/A'}")
        y-=18
    c.save()
    return filepath

def make_candidate_pdf(candidate, job_description):
    fname = f"candidate_{candidate['id']}_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
    filepath = os.path.join(app.config['REPORT_FOLDER'], fname)
    c = canvas.Canvas(filepath, pagesize=A4)
    width,height = A4
    x_margin,y = 40,height-50
    c.setFont("Helvetica-Bold",16)
    c.drawString(x_margin,y,f"Candidate Report — {candidate['id']}")
    y-=25
    c.setFont("Helvetica",10)
    c.drawString(x_margin,y,f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y-=18
    c.setFont("Helvetica-Bold",12); c.drawString(x_margin,y,"Summary")
    y-=14
    c.setFont("Helvetica",9)
    c.drawString(x_margin,y,f"Score: {candidate['score']} | Similarity: {candidate['sim_score']} | Coverage: {candidate['coverage']}%")
    y-=12
    c.drawString(x_margin,y,f"Matched Skills: {', '.join(candidate['matched_skills']) or 'None'}")
    y-=12
    c.drawString(x_margin,y,f"Missing Skills: {', '.join(candidate['missing_skills']) or 'None'}")
    y-=12
    c.drawString(x_margin,y,f"Experience: {candidate.get('years_experience',0)} years")
    y-=12
    c.drawString(x_margin,y,"Recommendations:")
    y-=12
    # training suggestions for missing skills
    recs = []
    for s in candidate['missing_skills']:
        ss = s.lower()
        if ss in TRAINING_MAP:
            recs.append(f"{s}: {', '.join(TRAINING_MAP[ss][:2])}")
    if not recs:
        c.drawString(x_margin,y,"No targeted training links available.")
        y-=12
    else:
        for r in recs:
            c.drawString(x_margin,y, "- " + r)
            y-=10
    y-=10
    c.setFont("Helvetica-Bold",11)
    c.drawString(x_margin,y,"Resume Excerpt (anonymized):")
    y-=12
    c.setFont("Helvetica",8)
    excerpt = candidate.get('text_excerpt','')[:1500]
    for chunk in re.findall('.{1,120}', excerpt):
        c.drawString(x_margin,y,chunk)
        y-=10
        if y<60:
            c.showPage()
            y=height-50
    c.save()
    return filepath

# ---- WORDCLOUD ----
def generate_wordcloud(all_skills, outpath):
    text = " ".join(all_skills)
    if not text.strip():
        text = "none"
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    wc.to_file(outpath)
    return outpath

# ---- CSV/EXCEL EXPORT ----
def export_results_to_csv(top_candidates):
    df = pd.DataFrame(top_candidates)
    fname = f"results_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
    path = os.path.join(app.config['REPORT_FOLDER'], fname)
    df.to_csv(path, index=False)
    return path

def export_results_to_excel(top_candidates):
    df = pd.DataFrame(top_candidates)
    fname = f"results_{datetime.now().strftime('%Y%m%d%H%M%S')}.xlsx"
    path = os.path.join(app.config['REPORT_FOLDER'], fname)
    df.to_excel(path, index=False)
    return path

# ---- ROUTES ----
@app.route('/login', methods=['GET','POST'])
def login():
    if request.method=='POST':
        pwd = request.form.get('password','')
        if pwd == ADMIN_PASS:
            session['admin'] = True
            flash('Logged in as admin', 'success')
            return redirect(url_for('matchresume'))
        else:
            flash('Incorrect password', 'danger')
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('admin', None)
    flash('Logged out', 'info')
    return redirect(url_for('matchresume'))

@app.route('/')
def matchresume():
    templates = {
        'Data Analyst': "Data Analyst JD templates... (python, sql, pandas, excel)",
        'Backend Developer': "Backend Developer JD templates... (python, flask/django, sql, rest api)",
        'Frontend Developer': "Frontend Developer JD templates... (html, css, javascript, react)"
    }
    logged_in = session.get('admin', False)
    return render_template('matchresume.html', job_templates=templates, logged_in=logged_in)

@app.route('/matcher', methods=['POST'])
def matcher():
    # get form inputs
    job_description = request.form.get('job_description','').strip()
    critical_skills_raw = request.form.get('critical_skills','').strip()
    critical_skills = [s.strip() for s in critical_skills_raw.split(',') if s.strip()]
    weight_similarity = float(request.form.get('weight_similarity',0.6))
    weight_skill_coverage = float(request.form.get('weight_skill_coverage',0.4))

    # handle uploads: either files or zip
    resume_files = request.files.getlist('resumes')
    zip_file = request.files.get('resume_zip')

    all_uploaded_paths = []
    saved_infos = []
    idx = 0

    def save_file_storage(fstorage):
        nonlocal idx
        idx += 1
        filename = secure_filename(fstorage.filename)
        uid = uuid.uuid4().hex
        outname = f"{uid}_{filename}"
        path = os.path.join(app.config['UPLOAD_FOLDER'], outname)
        fstorage.save(path)
        return path, filename

    # save single files
    for f in resume_files:
        if f and f.filename:
            _, ext = os.path.splitext(f.filename.lower())
            if ext in app.config['ALLOWED_EXT']:
                p, orig = save_file_storage(f)
                all_uploaded_paths.append((p, orig))
    # unzip zip
    if zip_file and zip_file.filename:
        try:
            mem = io.BytesIO(zip_file.read())
            with zipfile.ZipFile(mem) as z:
                for fn in z.namelist():
                    if fn.endswith('/'): continue
                    _, ext = os.path.splitext(fn.lower())
                    if ext in app.config['ALLOWED_EXT']:
                        data = z.read(fn)
                        uid = uuid.uuid4().hex
                        outname = f"{uid}_{os.path.basename(fn)}"
                        path = os.path.join(app.config['UPLOAD_FOLDER'], outname)
                        with open(path,'wb') as out:
                            out.write(data)
                        all_uploaded_paths.append((path, os.path.basename(fn)))
        except Exception as e:
            flash("Error extracting zip: " + str(e),'danger')

    if not job_description or not all_uploaded_paths:
        flash("Please provide Job Description and at least one resume file (or ZIP).",'warning')
        return redirect(url_for('matchresume'))

    resumes_texts = []
    resume_infos = []
    all_skills_collected = []
    for i, (path, orig) in enumerate(all_uploaded_paths, start=1):
        cid = f"RES{i}"
        raw_text = extract_text(path) or ""
        anon = anonymize_text(raw_text, cid)
        skills = extract_skills_from_text(anon)
        all_skills_collected.extend(skills)
        exp_edu = extract_experience_and_education(raw_text)
        resume_infos.append({
            'id': cid,
            'orig_filename': orig,
            'skills': skills,
            'years_experience': exp_edu.get('years_experience',0),
            'degrees': exp_edu.get('degrees',[]),
            'raw_text': anon
        })
        resumes_texts.append(anon)

    # scoring
    top_candidates, suggested_skills = compute_scores(job_description, resumes_texts, resume_infos,
                                                     critical_skills=critical_skills,
                                                     weight_similarity=weight_similarity,
                                                     weight_skill_coverage=weight_skill_coverage)

    # sort and top 10
    top_sorted = sorted(top_candidates, key=lambda x: x['score'], reverse=True)
    top_n = top_sorted[:10]

    # generate summary report and per-candidate reports (if admin logged in)
    report_name = f"report_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
    report_path = make_pdf_report(report_name, job_description, top_n)

    per_candidate_reports = []
    for c in top_n:
        p = make_candidate_pdf(c, job_description)
        per_candidate_reports.append({'id': c['id'], 'path': os.path.basename(p)})

    # generate wordcloud
    wc_path = os.path.join(app.config['STATIC_FOLDER'],'images', f"wc_{datetime.now().strftime('%Y%m%d%H%M%S')}.png")
    try:
        generate_wordcloud(all_skills_collected, wc_path)
    except Exception:
        wc_path = None

    # create exports
    csv_path = export_results_to_csv(top_n)
    excel_path = export_results_to_excel(top_n)

    # career suggestions: match candidate skills to roles
    for c in top_n:
        c_sk = set([s.lower() for s in c['matched_skills']])
        suggestions = []
        for role, skills in ROLE_MAP.items():
            score = len(c_sk.intersection(set(skills))) / max(len(skills),1)
            if score>0:
                suggestions.append((role, round(score,2)))
        suggestions = sorted(suggestions, key=lambda x: x[1], reverse=True)
        c['career_suggestions'] = suggestions[:3]

    # training suggestions for missing skills (first 3)
    for c in top_n:
        tr = []
        for miss in c['missing_skills'][:5]:
            if miss.lower() in TRAINING_MAP:
                tr.append({miss: TRAINING_MAP[miss.lower()]})
        c['training_suggestions'] = tr

    return render_template('matchresume.html',
                           message="Top matching resumes (anonymized):",
                           top_candidates=top_n,
                           suggested_skills=suggested_skills,
                           chart_scores=[c['score'] for c in top_n],
                           chart_labels=[{'id':c['id'],'file':c['orig_filename']} for c in top_n],
                           chart_coverages=[c['coverage'] for c in top_n],
                           report_file=os.path.basename(report_path),
                           candidate_reports=per_candidate_reports,
                           wordcloud=os.path.basename(wc_path) if wc_path else None,
                           csv_file=os.path.basename(csv_path),
                           excel_file=os.path.basename(excel_path),
                           job_description=job_description)

@app.route('/reports/<path:filename>')
def download_report(filename):
    return send_from_directory(app.config['REPORT_FOLDER'], filename, as_attachment=True)

@app.route('/static/images/<path:filename>')
def static_images(filename):
    return send_from_directory(os.path.join(app.config['STATIC_FOLDER'],'images'), filename)

@app.route('/download_candidate/<path:filename>')
def download_candidate_report(filename):
    return send_from_directory(app.config['REPORT_FOLDER'], filename, as_attachment=True)

# endpoint to download csv/excel
@app.route('/download_export/<path:filename>')
def download_export(filename):
    return send_from_directory(app.config['REPORT_FOLDER'], filename, as_attachment=True)

@app.route('/resume_maker', methods=['GET', 'POST'])
def resume_maker():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        phone = request.form.get('phone')
        education = request.form.get('education')
        experience = request.form.get('experience')
        skills = request.form.get('skills')

        # generate resume PDF
        filename = f"resume_{uuid.uuid4().hex}.pdf"
        filepath = os.path.join(app.config['REPORT_FOLDER'], filename)

        c = pdf_canvas.Canvas(filepath, pagesize=letter)
        width, height = letter
        y = height - 50

        c.setFont("Helvetica-Bold", 20)
        c.drawString(50, y, name)
        y -= 30

        c.setFont("Helvetica", 12)
        c.drawString(50, y, f"Email: {email}")
        y -= 20
        c.drawString(50, y, f"Phone: {phone}")
        y -= 40

        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y, "Education")
        y -= 20
        c.setFont("Helvetica", 12)
        for line in education.split("\n"):
            c.drawString(60, y, line.strip())
            y -= 20

        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y, "Experience")
        y -= 20
        c.setFont("Helvetica", 12)
        for line in experience.split("\n"):
            c.drawString(60, y, line.strip())
            y -= 20

        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y, "Skills")
        y -= 20
        c.setFont("Helvetica", 12)
        for skill in skills.split(","):
            c.drawString(60, y, skill.strip())
            y -= 20

        c.save()
        return send_file(filepath, as_attachment=True)

    return render_template("resume_maker.html")


if __name__=="__main__":
    app.run(debug=True)
