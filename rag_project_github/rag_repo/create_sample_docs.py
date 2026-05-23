from fpdf import FPDF

def create_pdf(filename, title, content_pages):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    for page_content in content_pages:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=11)
        pdf.ln(5)
        for para in page_content:
            pdf.multi_cell(0, 7, para)
            pdf.ln(3)
    pdf.output(filename)
    print(f"Created: {filename}")

# Document 1: Introduction to Machine Learning
ml_pages = [
    [
        "Introduction to Machine Learning",
        "Machine learning (ML) is a branch of artificial intelligence (AI) and computer science which focuses on the use of data and algorithms to imitate the way that humans learn, gradually improving its accuracy.",
        "Machine learning is an important component of the growing field of data science. Through the use of statistical methods, algorithms are trained to make classifications or predictions, uncovering key insights within data mining projects.",
        "There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.",
        "Supervised learning uses labeled training data to learn a mapping function from input variables to output variables. Common supervised learning algorithms include linear regression, logistic regression, decision trees, random forests, and neural networks.",
    ],
    [
        "Deep Learning and Neural Networks",
        "Deep learning is a subset of machine learning that uses multi-layered neural networks to analyze data. These neural networks are inspired by the structure of the human brain, with interconnected nodes or neurons that process information.",
        "Convolutional Neural Networks (CNNs) are particularly effective for image recognition tasks. They use convolutional layers to automatically detect features in images, such as edges, textures, and shapes.",
        "Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks are designed to work with sequential data such as time series or natural language. They maintain a hidden state that captures information about previous inputs.",
        "Transfer learning is a technique where a model trained on one task is reused as the starting point for a model on a different but related task. This significantly reduces training time and data requirements.",
    ],
    [
        "Applications of Machine Learning",
        "Machine learning has found applications across virtually every industry. In healthcare, ML models are used to diagnose diseases, predict patient outcomes, and discover new drugs. For example, convolutional neural networks can detect cancer from medical images with accuracy comparable to expert radiologists.",
        "In finance, machine learning is used for fraud detection, algorithmic trading, credit scoring, and risk assessment. Natural language processing models analyze financial news and reports to predict market movements.",
        "Self-driving cars rely heavily on machine learning for perception, decision-making, and control. These systems use cameras, lidar, and radar sensors combined with deep learning models to navigate complex environments.",
        "Natural Language Processing (NLP) enables machines to understand and generate human language. Applications include machine translation, sentiment analysis, chatbots, and document summarization.",
        "The global machine learning market was valued at $15.44 billion in 2021 and is expected to reach $209.91 billion by 2029, growing at a CAGR of 38.8%.",
    ]
]

# Document 2: Climate Change Report
climate_pages = [
    [
        "Global Climate Change: An Overview",
        "Climate change refers to long-term shifts in global temperatures and weather patterns. While climate change can occur naturally, scientific evidence overwhelmingly shows that human activities have been the main driver of climate change since the 1800s.",
        "The primary cause of modern climate change is the burning of fossil fuels such as coal, oil, and natural gas. This releases greenhouse gases, primarily carbon dioxide (CO2) and methane (CH4), which trap heat in the atmosphere.",
        "Global average temperatures have risen by approximately 1.1 degrees Celsius since pre-industrial times. This seemingly small increase has already caused significant disruptions to weather patterns, ecosystems, and human societies.",
        "The Intergovernmental Panel on Climate Change (IPCC) has stated that limiting global warming to 1.5°C above pre-industrial levels would significantly reduce the risks and impacts of climate change compared to warming of 2°C or higher.",
    ],
    [
        "Carbon Emissions and Their Impact",
        "Carbon dioxide emissions from fossil fuels reached a record high of 37.4 billion tonnes in 2023. The top five emitters are China (30%), the United States (14%), India (7%), Russia (5%), and Japan (3%).",
        "Deforestation accounts for approximately 10% of global carbon emissions. Forests absorb CO2 and act as carbon sinks, so their destruction both releases stored carbon and reduces the planet's capacity to absorb future emissions.",
        "The transportation sector is responsible for about 16% of global greenhouse gas emissions, with road vehicles accounting for the largest share. Aviation contributes approximately 2.5% of global CO2 emissions.",
        "Methane, while present in smaller quantities than CO2, is 80 times more potent as a greenhouse gas over a 20-year period. Agricultural activities, particularly livestock farming and rice cultivation, are major sources of methane emissions.",
    ],
    [
        "Renewable Energy and Solutions",
        "Solar power has become the cheapest source of electricity in history, with costs falling by 89% between 2010 and 2020. The global solar capacity reached 1,000 GW in 2022, up from just 40 GW in 2010.",
        "Wind energy capacity has grown dramatically, with global installed capacity reaching 837 GW by the end of 2022. Offshore wind energy is particularly promising due to stronger and more consistent wind speeds.",
        "Electric vehicles (EVs) are a key solution for reducing transportation emissions. Global EV sales reached 10.5 million in 2022, representing 14% of all new car sales. Battery costs have fallen by 97% since 1991.",
        "Carbon capture and storage (CCS) technology can remove CO2 from the atmosphere or capture it at the source of emission. However, current CCS capacity is limited and costs remain high at $50-100 per tonne of CO2.",
        "The global transition to renewable energy could create 30 million jobs by 2030, according to the International Renewable Energy Agency (IRENA), offsetting losses in fossil fuel industries.",
    ]
]

# Document 3: Artificial Intelligence Ethics
ai_ethics_pages = [
    [
        "Ethical Considerations in Artificial Intelligence",
        "Artificial intelligence ethics is a set of values, principles, and techniques that employ widely accepted standards of right and wrong to guide moral conduct in the development and use of AI technologies.",
        "Bias in AI systems is a critical concern. When training data reflects historical inequalities or societal biases, AI models can perpetuate and even amplify these biases. For example, facial recognition systems have shown significantly higher error rates for women and people with darker skin tones.",
        "Transparency and explainability are essential for trustworthy AI. Many deep learning models operate as 'black boxes', making decisions that cannot be easily explained or interpreted. This lack of transparency raises concerns in high-stakes applications like healthcare and criminal justice.",
        "Privacy is a major issue in AI development. AI systems often require vast amounts of personal data for training, raising questions about consent, data ownership, and the potential for surveillance.",
    ],
    [
        "AI Governance and Regulation",
        "The European Union's Artificial Intelligence Act, adopted in 2024, is the world's first comprehensive AI regulation. It classifies AI systems by risk level and imposes strict requirements on high-risk applications.",
        "The AI Act prohibits certain uses of AI deemed unacceptable, such as social scoring by governments and real-time biometric surveillance in public spaces. High-risk AI systems, including those used in critical infrastructure and education, must meet stringent requirements for transparency and human oversight.",
        "In the United States, AI regulation has been more fragmented, with sector-specific guidelines rather than comprehensive legislation. Executive Order 14110, signed in October 2023, established new standards for AI safety and security.",
        "China has implemented a series of AI regulations, including measures governing recommendation algorithms, deep synthesis technologies (deepfakes), and generative AI services. These regulations require AI systems to respect national values and prevent the generation of false information.",
    ],
    [
        "The Future of AI Ethics",
        "Autonomous weapons systems, sometimes called 'killer robots', represent one of the most controversial applications of AI. Critics argue that delegating life-and-death decisions to machines violates fundamental ethical principles.",
        "The development of artificial general intelligence (AGI) - AI that matches or surpasses human cognitive abilities across all domains - raises profound ethical questions about control, alignment with human values, and the long-term future of humanity.",
        "AI alignment research focuses on ensuring that advanced AI systems behave in accordance with human values and intentions. This includes technical challenges such as specifying reward functions and interpretability, as well as philosophical questions about whose values should be encoded.",
        "Inclusive AI development requires diverse teams and perspectives to prevent the encoding of narrow viewpoints into powerful systems. Studies show that teams with greater diversity produce more robust and less biased AI systems.",
        "Carbon emissions from machine learning have become a growing concern. Training large language models can emit as much carbon as five cars over their lifetimes. The AI industry must prioritize energy efficiency and renewable energy.",
    ]
]

docs_dir = "/home/claude/rag_project/docs"
create_pdf(f"{docs_dir}/machine_learning_guide.pdf", "Machine Learning Guide", ml_pages)
create_pdf(f"{docs_dir}/climate_change_report.pdf", "Climate Change Report", climate_pages)
create_pdf(f"{docs_dir}/ai_ethics_guide.pdf", "AI Ethics Guide", ai_ethics_pages)

print("\nAll sample documents created successfully!")
