from __future__ import annotations

from typing import Dict, List

# Add/remove skills as you like. Keys are canonical skill names; values are aliases to match in text.
SKILL_LEXICON: Dict[str, List[str]] = {
    # ======================
    # Languages
    # ======================
    "java": ["java"],
    "python": ["python"],
    "javascript": ["javascript", "js", "ecmascript", "es6"],
    "typescript": ["typescript", "ts"],
    "php": ["php"],
    "kotlin": ["kotlin"],
    "swift": ["swift"],
    "c": ["c programming", "c"],
    "c++": ["c++", "cpp"],
    "c#": ["c#", "c sharp", "c-sharp", "c# .net", "c#.net"],
    "go": ["go", "golang"],
    "rust": ["rust"],
    "ruby": ["ruby"],
    "scala": ["scala"],
    "r": ["r language", "r programming"],
    "bash": ["bash", "shell scripting"],
    "powershell": ["powershell"],

    # ======================
    # Frontend
    # ======================
    "react": ["react", "react.js", "reactjs"],
    "react native": ["react native", "react-native", "reactnative"],
    "next.js": ["next.js", "nextjs", "next js"],
    "vue": ["vue", "vue.js", "vuejs"],
    "nuxt": ["nuxt", "nuxt.js", "nuxtjs"],
    "angular": ["angular", "angularjs", "angluar"],
    "flutter": ["flutter"],
    "html": ["html", "html5"],
    "css": ["css", "css3"],
    "sass": ["sass", "scss"],
    "tailwind": ["tailwind", "tailwind css"],
    "bootstrap": ["bootstrap"],
    "storybook": ["storybook"],
    "three.js": ["three.js", "threejs"],
    "webgl": ["webgl"],

    # ======================
    # State / Frontend Tools
    # ======================
    "redux": ["redux", "redux toolkit"],
    "zustand": ["zustand"],
    "recoil": ["recoil"],
    "react query": ["react query", "tanstack query"],
    "swr": ["swr"],
    "webpack": ["webpack"],
    "vite": ["vite"],
    "babel": ["babel"],
    "eslint": ["eslint"],
    "prettier": ["prettier"],

    # ======================
    # Testing
    # ======================
    "jest": ["jest"],
    "react testing library": ["react testing library", "rtl"],
    "cypress": ["cypress"],
    "playwright": ["playwright"],
    "selenium": ["selenium"],
    "pytest": ["pytest"],
    "junit": ["junit", "j unit"],
    "oop": ["oop", "oop concepts", "object oriented programming", "object-oriented programming"],
    "tdd": ["tdd", "test driven development"],
    "bdd": ["bdd", "behavior driven development"],

    # ======================
    # Backend / APIs
    # ======================
    "node.js": ["node.js", "nodejs", "node"],
    "express": ["express", "express.js"],
    "nestjs": ["nestjs", "nest.js"],
    "django": ["django"],
    "flask": ["flask"],
    "fastapi": ["fastapi"],
    "spring framework": ["spring framework", "spring"],
    "spring boot": ["spring boot"],
    "java ee": ["java ee", "j2ee", "jee"],
    ".net": [".net", "dotnet", "asp.net", "aspnet"],
    "graphql": ["graphql"],
    "grpc": ["grpc"],
    "rest api": [
        "rest",
        "restful",
        "rest api",
        "restful api",
        "rest apis",
        "api integration",
        "api integrations",
    ],
    "api gateway": ["api gateway"],
    "web services": ["web services", "soap", "wsdl"],
    "webhooks": ["webhook", "webhooks"],
    "oauth": ["oauth", "oauth2", "o auth"],
    "jwt": ["jwt", "json web token", "json web tokens"],

    # ======================
    # Databases
    # ======================
    "sql": ["sql"],
    "pl/sql": ["pl/sql", "pl sql", "plsql"],
    "mysql": ["mysql"],
    "postgresql": ["postgresql", "postgres"],
    "oracle": ["oracle", "oracle db", "rdbms oracle", "rdbms"],
    "sql server": ["sql server", "mssql"],
    "mongodb": ["mongodb", "mongo db", "mongo"],
    "redis": ["redis"],
    "cassandra": ["cassandra"],
    "dynamodb": ["dynamodb"],
    "firebase": ["firebase", "firestore"],
    "elasticsearch": ["elasticsearch", "elastic search"],
    "neo4j": ["neo4j", "graph database"],
    "vector database": ["vector database", "pinecone", "weaviate", "milvus"],

    # ======================
    # Data / AI / ML
    # ======================
    "machine learning": ["machine learning", "ml"],
    "deep learning": ["deep learning", "dl"],
    "nlp": ["nlp", "natural language processing"],
    "computer vision": ["computer vision"],
    "llm": ["llm", "large language model", "large language models"],
    "rag": ["rag", "retrieval augmented generation"],
    "tensorflow": ["tensorflow"],
    "pytorch": ["pytorch"],
    "scikit-learn": ["scikit learn", "sklearn"],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "hugging face": ["hugging face", "transformers"],
    "prompt engineering": ["prompt engineering"],

    # ======================
    # Cloud & DevOps
    # ======================
    "aws": ["aws", "amazon web services"],
    "gcp": ["gcp", "google cloud", "google cloud platform"],
    "azure": ["azure", "microsoft azure"],
    "serverless": ["serverless", "lambda", "cloud functions"],
    "docker": ["docker"],
    "kubernetes": ["kubernetes", "k8s"],
    "helm": ["helm"],
    "terraform": ["terraform"],
    "ansible": ["ansible"],
    "ci/cd": ["ci/cd", "cicd", "pipelines"],
    "github actions": ["github actions", "github action"],
    "gitlab ci": ["gitlab ci"],
    "jenkins": ["jenkins"],
    "git": ["git"],
    "linux": ["linux"],
    "monitoring": ["monitoring", "observability"],
    "prometheus": ["prometheus"],
    "grafana": ["grafana"],

    # ======================
    # Mobile
    # ======================
    "mobile": ["mobile", "mobile app", "mobile apps", "mobile development"],
    "android": ["android", "android development"],
    "ios": ["ios", "ios development"],
    "xamarin": ["xamarin"],

    # ======================
    # Security & QA
    # ======================
    "cybersecurity": ["cybersecurity", "information security"],
    "penetration testing": ["penetration testing", "pentesting"],
    "owasp": ["owasp", "owasp top 10"],
    "devsecops": ["devsecops"],
    "qa": ["qa", "quality assurance"],
    "automation testing": ["automation testing"],
    "crystal reports": ["crystal reports", "crystal report"],

    # ======================
    # Automation/Enterprise
    # ======================
    "rpa": ["rpa", "robotic process automation"],
    "uipath": ["uipath", "ui-path", "ui path"],
    "sap": ["sap"],
    "salesforce": ["salesforce"],
    "servicenow": ["servicenow"],
}

# Some common "soft requirements" you may want to track (optional)
EXTRA_SIGNALS = {
    "full stack": ["full stack", "full-stack"],
    "backend": ["backend", "back-end"],
    "frontend": ["frontend", "front-end"],
    "microservices": ["microservices", "micro-service"],
    "event driven": ["event driven", "event-driven"],
    "integration": ["integration", "integrations"],
    "spa": ["single page", "spa", "single-page"],
    "ssr": ["server-side rendering", "ssr"],
    "cloud native": ["cloud native"],
    "scalable systems": ["scalable", "highly scalable"],
}

SECTION_HEADERS = [
    "responsibilities",
    "requirements",
    "must have",
    "must-have",
    "mandatory requirements",
    "core skills",
    "core skills & experience",
    "technical skills",
    "preferred qualifications",
    "preferred skills",
    "what you will do",
    "what you bring",
    "nice-to-have",
    "nice to have",
    "good to have",
    "bonus skills",
    "the job",
    "the person",
]
