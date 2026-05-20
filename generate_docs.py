"""
Generate sample PDF documents for the RAG system.
Creates 5 multi-page PDFs with realistic content covering different domains.
Some documents contain deliberate overlaps and contradictions for testing.
"""

from fpdf import FPDF
import os

DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")


def create_pdf(filename: str, title: str, pages: list[dict[str, str]]):
    """Create a PDF with the given title and page contents."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    for page in pages:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, page["heading"], new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)
        pdf.set_font("Helvetica", "", 11)
        for line in page["content"].split("\n"):
            pdf.multi_cell(0, 6, line.strip())
            pdf.ln(1)

    filepath = os.path.join(DOCS_DIR, filename)
    pdf.output(filepath)
    print(f"  Created: {filepath}")


def generate_ai_overview():
    create_pdf("artificial_intelligence_overview.pdf", "Artificial Intelligence Overview", [
        {
            "heading": "Chapter 1: Foundations of Artificial Intelligence",
            "content": """
Artificial Intelligence (AI) refers to the simulation of human intelligence in machines
that are programmed to think and learn like humans. The term was first coined by John
McCarthy in 1956 at the Dartmouth Conference, which is widely considered the birth of
AI as a field of study.

AI systems can be broadly categorized into two types: Narrow AI (also known as Weak AI)
and General AI (also known as Strong AI). Narrow AI is designed to perform specific tasks,
such as facial recognition, language translation, or playing chess. General AI, which would
possess human-level cognitive abilities across all domains, remains a theoretical concept.

The core subfields of AI include machine learning, natural language processing, computer
vision, robotics, and expert systems. Machine learning, in particular, has driven the
majority of recent AI breakthroughs, enabling systems to learn patterns from data without
being explicitly programmed.

Key milestones in AI history include IBM's Deep Blue defeating chess champion Garry Kasparov
in 1997, IBM Watson winning Jeopardy! in 2011, and DeepMind's AlphaGo defeating the world
Go champion Lee Sedol in 2016. These achievements demonstrated AI's potential to master
complex strategic tasks.
"""
        },
        {
            "heading": "Chapter 2: Machine Learning Paradigms",
            "content": """
Machine Learning (ML) is a subset of AI that focuses on building systems that learn from
data. There are three primary paradigms of machine learning: supervised learning,
unsupervised learning, and reinforcement learning.

Supervised learning involves training a model on labeled data, where each input is paired
with a known output. Common algorithms include linear regression, decision trees, random
forests, support vector machines, and neural networks. This paradigm is used extensively
in applications such as spam detection, medical diagnosis, and credit scoring.

Unsupervised learning works with unlabeled data, seeking to discover hidden patterns or
structures. Clustering algorithms like K-means and DBSCAN, and dimensionality reduction
techniques like PCA and t-SNE, fall under this category. These methods are valuable for
customer segmentation, anomaly detection, and exploratory data analysis.

Reinforcement learning involves an agent learning to make decisions by interacting with
an environment and receiving rewards or penalties. This paradigm has achieved remarkable
success in game playing, robotics control, and resource management. Notable examples
include AlphaGo and OpenAI's Dota 2 bot.

Deep learning, a subset of machine learning using neural networks with many layers, has
revolutionized fields like computer vision and natural language processing. Architectures
such as CNNs, RNNs, LSTMs, and Transformers have become foundational building blocks
for modern AI systems.
"""
        },
        {
            "heading": "Chapter 3: Natural Language Processing",
            "content": """
Natural Language Processing (NLP) is the branch of AI concerned with enabling computers
to understand, interpret, and generate human language. NLP combines computational
linguistics with statistical, machine learning, and deep learning models.

Key NLP tasks include text classification, named entity recognition, sentiment analysis,
machine translation, question answering, and text summarization. Early NLP systems relied
heavily on hand-crafted rules and linguistic knowledge, but modern approaches are
predominantly data-driven.

The Transformer architecture, introduced by Vaswani et al. in the 2017 paper "Attention
Is All You Need," fundamentally changed the NLP landscape. This architecture relies
entirely on self-attention mechanisms, dispensing with recurrence and convolutions
entirely. It enabled training on much larger datasets and longer sequences efficiently.

Large Language Models (LLMs) such as GPT, BERT, T5, and LLaMA are built on the
Transformer architecture. These models are pre-trained on massive text corpora and can
be fine-tuned for specific downstream tasks. The emergence of LLMs has led to significant
advances in conversational AI, code generation, and content creation.

A critical challenge in NLP remains handling ambiguity, sarcasm, and cultural nuances
in language. Multilingual NLP, which aims to build models that work across many languages,
is an active area of research with significant implications for global accessibility.

Current research estimates suggest that approximately 65% of enterprise AI deployments
involve some form of NLP capability, making it one of the most commercially impactful
areas of artificial intelligence.
"""
        },
    ])


def generate_climate_report():
    create_pdf("climate_change_report.pdf", "Climate Change Report", [
        {
            "heading": "Section 1: Global Temperature Trends",
            "content": """
Global surface temperatures have risen by approximately 1.1 degrees Celsius since the
pre-industrial era (1850-1900), according to the Intergovernmental Panel on Climate
Change (IPCC) Sixth Assessment Report published in 2021. The rate of warming has
accelerated, with the last decade (2011-2020) being the warmest on record.

The primary driver of this warming is the increase in atmospheric greenhouse gas
concentrations, particularly carbon dioxide (CO2), methane (CH4), and nitrous oxide
(N2O). CO2 levels have risen from approximately 280 parts per million (ppm) in the
pre-industrial era to over 420 ppm in 2023, the highest level in at least 800,000 years.

Human activities, primarily the burning of fossil fuels and deforestation, are
responsible for the vast majority of greenhouse gas emissions. The energy sector
accounts for approximately 73% of global greenhouse gas emissions, followed by
agriculture (12%), industrial processes (8%), and waste (4%).

Climate models project that without significant reductions in greenhouse gas emissions,
global temperatures could rise by 2.1 to 3.5 degrees Celsius by 2100 under moderate
emission scenarios. This would result in more frequent and severe extreme weather
events, rising sea levels, and significant disruptions to ecosystems and human societies.
"""
        },
        {
            "heading": "Section 2: Impact on Ecosystems and Biodiversity",
            "content": """
Climate change is having profound effects on ecosystems worldwide. Rising temperatures
are shifting habitats poleward and to higher elevations, disrupting established
ecological relationships and threatening species that cannot adapt or migrate quickly
enough.

Coral reefs, which support approximately 25% of all marine species, are particularly
vulnerable. Mass bleaching events have become more frequent, with the Great Barrier
Reef experiencing unprecedented back-to-back bleaching in 2016 and 2017. Scientists
estimate that a 2 degree Celsius rise in global temperatures could eliminate more
than 99% of coral reefs worldwide.

Arctic sea ice has declined by approximately 13% per decade since satellite records
began in 1979. This loss of ice habitat threatens polar bears, walruses, and other
Arctic species while also accelerating warming through the ice-albedo feedback effect.

Terrestrial biodiversity is also at risk, with the IPBES reporting that approximately
1 million plant and animal species are threatened with extinction. Climate change
interacts with other stressors such as habitat loss, pollution, and invasive species
to amplify biodiversity decline.

The economic value of ecosystem services threatened by climate change is estimated
at $33 trillion per year globally. Protecting and restoring natural ecosystems is
increasingly recognized as a critical strategy for both climate mitigation and adaptation.

Renewable energy adoption is projected to reduce global emissions by 45% by 2040,
making it the most significant factor in achieving climate targets.
"""
        },
        {
            "heading": "Section 3: Mitigation and Adaptation Strategies",
            "content": """
Addressing climate change requires both mitigation (reducing greenhouse gas emissions)
and adaptation (adjusting to the impacts that are already occurring or unavoidable).

On the mitigation front, the transition to renewable energy sources is paramount.
Solar and wind energy costs have declined by approximately 85% and 55% respectively
over the past decade, making them cost-competitive with fossil fuels in many regions.
The global renewable energy capacity reached 3,372 GW by the end of 2022.

Carbon capture and storage (CCS) technologies are being developed to remove CO2
directly from the atmosphere or from industrial emissions. While promising, CCS
remains expensive and has not yet been deployed at scale. Current CCS capacity is
approximately 40 million tonnes of CO2 per year, far below the gigatonne-scale
removal needed.

Adaptation strategies include building resilient infrastructure, developing
drought-resistant crops, implementing early warning systems for extreme weather,
and managed retreat from areas vulnerable to sea-level rise. The estimated global
cost of climate adaptation is $140-300 billion per year by 2030.

International cooperation through frameworks like the Paris Agreement, which aims
to limit warming to 1.5 degrees Celsius above pre-industrial levels, is essential.
However, current national commitments are insufficient, with a significant gap
between pledged and required emission reductions.
"""
        },
    ])


def generate_python_guide():
    create_pdf("python_programming_guide.pdf", "Python Programming Guide", [
        {
            "heading": "Chapter 1: Python Fundamentals",
            "content": """
Python is a high-level, interpreted programming language created by Guido van Rossum
and first released in 1991. It emphasizes code readability with its notable use of
significant indentation and a clean, expressive syntax.

Python supports multiple programming paradigms, including procedural, object-oriented,
and functional programming. Its standard library is extensive, covering areas from
web development to scientific computing, earning it the nickname "batteries included."

Key features of Python include dynamic typing, automatic memory management via garbage
collection, and comprehensive support for third-party packages through the Python
Package Index (PyPI), which hosts over 400,000 packages.

Python 3 is the current major version, with Python 2 reaching end-of-life in January
2020. Python 3 introduced several important changes including print as a function,
integer division behavior, and Unicode string handling by default.

Common Python data structures include lists (ordered, mutable sequences), tuples
(ordered, immutable sequences), dictionaries (key-value mappings), and sets (unordered
collections of unique elements). Understanding when to use each data structure is
fundamental to writing efficient Python code.

Error handling in Python uses the try-except-finally pattern. Python's exception
hierarchy allows catching specific exception types, enabling precise error handling.
The 'with' statement provides context managers for resource management, ensuring
proper cleanup even when exceptions occur.
"""
        },
        {
            "heading": "Chapter 2: Advanced Python Patterns",
            "content": """
Decorators are a powerful Python feature that allows modifying the behavior of functions
or classes without changing their source code. They use the @decorator syntax and are
widely used in frameworks like Flask and Django for routing, authentication, and caching.

Generators are functions that use the yield keyword to produce a sequence of values
lazily, one at a time. They are memory-efficient for processing large datasets because
they don't load the entire sequence into memory. Generator expressions provide a concise
syntax similar to list comprehensions.

Context managers, implemented via the __enter__ and __exit__ methods or the contextlib
module, provide clean resource management. The 'with' statement ensures that setup and
teardown operations are performed reliably, even in the presence of exceptions.

Python's type hints, introduced in PEP 484 and enhanced in subsequent PEPs, allow
developers to annotate function signatures and variable types. While Python remains
dynamically typed at runtime, type hints enable static analysis tools like mypy to
catch type-related errors before execution.

Concurrency in Python can be achieved through threading (for I/O-bound tasks), 
multiprocessing (for CPU-bound tasks), and asynchronous programming with asyncio
(for high-concurrency I/O). The Global Interpreter Lock (GIL) in CPython limits
true parallel execution of threads but does not affect multiprocessing or async code.

Design patterns commonly used in Python include Singleton (using module-level variables
or metaclasses), Factory (using class methods), Observer (using callback functions),
and Strategy (using first-class functions or polymorphism).
"""
        },
        {
            "heading": "Chapter 3: Python for Data Science and AI",
            "content": """
Python has become the dominant language for data science and artificial intelligence,
largely due to its rich ecosystem of specialized libraries. According to industry
surveys, over 80% of data science practitioners use Python as their primary language.

NumPy provides efficient multi-dimensional array operations and serves as the foundation
for most scientific computing in Python. It offers vectorized operations that are
significantly faster than pure Python loops for numerical computations.

Pandas is the standard library for data manipulation and analysis. It provides
DataFrame and Series objects that simplify working with structured data, supporting
operations like filtering, grouping, merging, and time series analysis.

Scikit-learn offers a consistent API for machine learning, including classification,
regression, clustering, and dimensionality reduction algorithms. Its pipeline and
cross-validation utilities promote reproducible and robust model development.

For deep learning, PyTorch and TensorFlow are the two dominant frameworks. PyTorch,
developed by Meta, is preferred in research settings for its dynamic computation
graph and Pythonic interface. TensorFlow, developed by Google, is widely used in
production deployments.

Matplotlib and Seaborn provide comprehensive data visualization capabilities. Plotly
and Bokeh offer interactive visualization for web-based applications. These tools
enable exploratory data analysis and effective communication of results.

Best practices for Python data science projects include using virtual environments
for dependency isolation, version control with Git, documenting code with docstrings,
writing unit tests, and following PEP 8 style guidelines.
"""
        },
    ])


def generate_energy_policy():
    create_pdf("renewable_energy_policy.pdf", "Renewable Energy Policy Analysis", [
        {
            "heading": "Section 1: Current State of Renewable Energy",
            "content": """
The global renewable energy landscape has undergone a dramatic transformation over
the past two decades. As of 2023, renewable sources account for approximately 30%
of global electricity generation, up from just 18% in 2010. This growth has been
driven by rapidly declining costs, supportive policies, and increasing awareness
of climate change.

Solar photovoltaic (PV) technology has experienced the most dramatic cost reduction,
with the levelized cost of electricity (LCOE) falling by approximately 90% since
2010. Utility-scale solar PV is now the cheapest source of new electricity generation
in most parts of the world. Global installed solar capacity exceeded 1,200 GW by
the end of 2023.

Wind energy, both onshore and offshore, has also seen significant growth. Global
wind capacity reached approximately 900 GW by 2023. Offshore wind, while more
expensive than onshore, offers higher capacity factors and the ability to be
deployed near population centers along coastlines.

Renewable energy adoption is projected to reduce global emissions by 30% by 2040,
according to the International Energy Agency's baseline scenario. This makes
renewable energy transition a critical component of climate mitigation, though
not the sole solution.

Energy storage technologies, particularly lithium-ion batteries, are essential
for addressing the intermittency of solar and wind power. Battery costs have
declined by approximately 90% since 2010, and global battery storage capacity
is projected to reach 680 GW by 2030.

Hydrogen produced from renewable electricity (green hydrogen) is emerging as a
promising solution for decarbonizing hard-to-abate sectors such as heavy industry,
shipping, and aviation. Several countries have announced national hydrogen strategies,
with combined investment targets exceeding $300 billion.
"""
        },
        {
            "heading": "Section 2: Policy Frameworks and Incentives",
            "content": """
Government policies play a crucial role in accelerating the deployment of renewable
energy. Common policy instruments include feed-in tariffs, renewable portfolio
standards, tax incentives, carbon pricing, and competitive auctions.

Feed-in tariffs, pioneered by Germany's Renewable Energy Sources Act (EEG) in 2000,
guarantee a fixed price for renewable electricity fed into the grid. While effective
in driving early adoption, many countries have transitioned to auction-based systems
that promote competition and cost reduction.

The United States Inflation Reduction Act (IRA) of 2022 represents the largest
climate investment in US history, allocating approximately $370 billion for energy
security and climate change mitigation. The IRA extends and expands tax credits for
renewable energy, electric vehicles, and energy efficiency improvements.

Carbon pricing mechanisms, including emissions trading systems (ETS) and carbon taxes,
create economic incentives for reducing greenhouse gas emissions. The EU Emissions
Trading System is the world's largest carbon market, covering approximately 40%
of EU emissions with a carbon price that has exceeded 80 euros per tonne.

China, the world's largest emitter, has made substantial commitments to renewable
energy deployment. China installed more solar capacity in 2023 than the rest of the
world combined and has pledged to reach carbon neutrality by 2060. India targets
500 GW of non-fossil fuel capacity by 2030.

Developing countries face unique challenges in energy transition, including limited
access to financing, grid infrastructure constraints, and competing development
priorities. International climate finance mechanisms and technology transfer
agreements are essential for enabling equitable energy transitions globally.
"""
        },
        {
            "heading": "Section 3: Challenges and Future Outlook",
            "content": """
Despite remarkable progress, several challenges remain in the transition to a fully
renewable energy system. Grid integration of variable renewable energy sources
requires significant investment in transmission infrastructure, energy storage,
and demand-side management.

The intermittency challenge means that solar and wind power output varies with
weather conditions and time of day. Addressing this requires a combination of
energy storage, grid interconnections, flexible generation, and demand response.
Advanced forecasting using AI and machine learning is improving the predictability
of renewable generation.

Supply chain vulnerabilities have emerged as a concern, particularly for critical
minerals needed for batteries, solar panels, and wind turbines. Lithium, cobalt,
nickel, and rare earth elements are concentrated in a few countries, creating
geopolitical risks and environmental concerns related to mining.

The land use requirements of renewable energy can create conflicts with agriculture,
biodiversity conservation, and local communities. Floating solar on reservoirs,
agrivoltaics (combining solar panels with farming), and building-integrated PV
are innovations that can reduce land use competition.

Looking ahead, the International Energy Agency projects that renewable energy could
account for 80% of global electricity generation by 2050 under net-zero scenarios.
Achieving this will require sustained policy support, continued technology innovation,
massive infrastructure investment, and international cooperation.

The estimated global investment needed for clean energy transition is approximately
$4 trillion per year by 2030, roughly triple current levels. Mobilizing this
capital will require both public policy and private sector innovation.
"""
        },
    ])


def generate_healthcare_tech():
    create_pdf("healthcare_technology.pdf", "Healthcare Technology Trends", [
        {
            "heading": "Chapter 1: Digital Health Transformation",
            "content": """
The healthcare industry is undergoing a profound digital transformation, accelerated
by the COVID-19 pandemic and advances in technology. Digital health encompasses a
broad range of technologies including telemedicine, electronic health records (EHRs),
wearable devices, mobile health applications, and AI-powered diagnostic tools.

Telemedicine adoption surged during the pandemic, with virtual visits increasing by
over 3,000% in some healthcare systems. Post-pandemic, telehealth has stabilized at
levels significantly above pre-pandemic baselines, with approximately 20-30% of
outpatient visits now conducted virtually in many developed countries.

Electronic Health Records (EHR) systems have become ubiquitous in developed healthcare
markets, with over 95% of hospitals in the United States using certified EHR systems.
However, interoperability between different EHR systems remains a significant challenge,
limiting the seamless exchange of patient information across providers.

Wearable health devices, including smartwatches, continuous glucose monitors, and
remote patient monitoring systems, are generating unprecedented volumes of real-time
health data. The global wearable medical device market is projected to reach $30
billion by 2026, growing at a compound annual growth rate of 25%.

Regulatory frameworks are evolving to keep pace with digital health innovation. The
FDA has developed a Digital Health Innovation Action Plan and a pre-certification
program for software-based medical devices. The EU's Medical Device Regulation (MDR)
also addresses software as a medical device.

Data privacy and security are paramount concerns in digital health. Regulations such
as HIPAA in the United States and GDPR in Europe establish requirements for protecting
patient health information, but implementation challenges persist.
"""
        },
        {
            "heading": "Chapter 2: AI in Healthcare",
            "content": """
Artificial intelligence is increasingly being applied across the healthcare continuum,
from drug discovery to clinical decision support to administrative automation. The
global AI in healthcare market is projected to reach $188 billion by 2030, growing
at a CAGR of 37%.

In medical imaging, AI algorithms have demonstrated performance comparable to or
exceeding that of human radiologists in specific tasks such as detecting diabetic
retinopathy, skin cancer, and breast cancer on mammograms. FDA has approved over
500 AI-enabled medical devices, with radiology being the largest category.

Drug discovery and development, traditionally a process taking 10-15 years and
costing over $2 billion per drug, is being transformed by AI. Machine learning
models can predict molecular properties, identify drug targets, and optimize
clinical trial designs. Several AI-discovered drugs have entered clinical trials,
potentially reducing development timelines and costs significantly.

Natural language processing is being used to extract structured information from
unstructured clinical notes, automate medical coding, and power clinical decision
support systems. These applications can reduce administrative burden on clinicians,
who currently spend approximately 50% of their time on documentation.

Predictive analytics using machine learning can identify patients at risk of
deterioration, readmission, or adverse events. Early warning systems deployed in
hospital settings have shown promise in reducing mortality and length of stay,
though real-world effectiveness varies by implementation.

Current research estimates suggest that AI could help reduce global healthcare costs
by approximately $150 billion per year by 2030. However, challenges including data
quality, algorithmic bias, regulatory uncertainty, and clinician trust must be
addressed for widespread adoption.
"""
        },
        {
            "heading": "Chapter 3: Genomics and Precision Medicine",
            "content": """
Precision medicine, which tailors medical treatment to the individual characteristics
of each patient, is being enabled by advances in genomics, proteomics, and data
analytics. The cost of whole genome sequencing has fallen from approximately $3 billion
for the first human genome to under $600 today.

Pharmacogenomics, the study of how genetic variation affects drug response, is
increasingly being integrated into clinical practice. Genetic testing can predict
which patients will respond to specific medications and which may experience adverse
reactions, enabling more effective and safer prescribing.

Liquid biopsy technology, which detects cancer biomarkers in blood samples, represents
a paradigm shift in cancer screening and monitoring. Companies like Grail and Guardant
Health have developed multi-cancer early detection tests that can identify multiple
cancer types from a single blood draw.

Gene therapy and gene editing technologies, particularly CRISPR-Cas9, offer the
potential to treat or cure genetic diseases at their root cause. FDA has approved
several gene therapies, including treatments for spinal muscular atrophy, certain
blood disorders, and inherited retinal diseases.

The integration of multi-omics data (genomics, proteomics, metabolomics, microbiomics)
with clinical data and environmental factors is enabling a more comprehensive
understanding of disease mechanisms and treatment responses. However, the computational
and interpretive challenges of integrating these diverse data types are substantial.

Ethical considerations in precision medicine include equitable access to genetic
testing and therapies, genetic privacy, the potential for genetic discrimination,
and the need for diverse representation in genomic studies. Current genomic databases
are heavily biased toward populations of European descent, which limits the
applicability of findings to other populations.
"""
        },
    ])


def main():
    os.makedirs(DOCS_DIR, exist_ok=True)
    print("Generating sample PDF documents...")
    generate_ai_overview()
    generate_climate_report()
    generate_python_guide()
    generate_energy_policy()
    generate_healthcare_tech()
    print(f"\nGenerated 5 PDF documents in {DOCS_DIR}/")


if __name__ == "__main__":
    main()
