import io
from pypdf import PdfReader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun


def extract_text_from_cv(uploaded_file) -> str:
    """Extracts plain text from an uploaded PDF resume/CV."""
    pdf_bytes = uploaded_file.read()
    uploaded_file.seek(0)  # Reset pointer so vectorstore can also read it if needed
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text_chunks = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(text_chunks).strip()


def web_search_context(query: str) -> str:
    """Performs a lightweight web search to enrich LLM answers or CV questions."""
    try:
        search = DuckDuckGoSearchRun()
        return search.run(query)
    except Exception as e:
        return f"Web search unavailable ({e}). Proceeding with LLM internal knowledge."


def generate_cv_questionnaire(
    llm,
    cv_text: str,
    target_role: str = "",
    use_web_search: bool = True,
    num_questions: int = 8,
) -> str:
    """
    Analyzes CV text and generates a structured interview/evaluation questionnaire,
    optionally enriched with live web search trends for the candidate's tech stack/role.
    """
    search_insights = "Not requested."
    if use_web_search:
        search_query = (
            f"latest technical interview questions trends {target_role}"
            if target_role
            else "latest technical and behavioral interview questions software engineering data science"
        )
        search_insights = web_search_context(search_query)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert technical recruiter and domain interviewer. "
                "Analyze the candidate's CV and generate a high-signal, tailored questionnaire. "
                "Format the output cleanly in Markdown with sections:\n"
                "1. **Candidate Snapshot** (2-3 bullet summary of core skills & experience)\n"
                "2. **Role-Specific Technical Questions** (deep-dive into projects/skills on the CV)\n"
                "3. **Scenario & Problem-Solving Questions** (incorporating industry trends)\n"
                "4. **Behavioral & Leadership Questions**\n"
                "For each question, include a brief **What to look for in a good answer** hint.",
            ),
            (
                "human",
                "Target Role (if specified): {target_role}\n"
                "Number of Questions: {num_questions}\n\n"
                "Recent Industry / Search Context:\n{search_insights}\n\n"
                "Candidate CV Content:\n{cv_text}",
            ),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    return chain.invoke(
        {
            "target_role": target_role or "Infer best-fit role from CV",
            "num_questions": num_questions,
            "search_insights": search_insights,
            "cv_text": cv_text[:12000],  # Guard against token overflow
        }
    )