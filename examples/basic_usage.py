"""
Basic usage examples for EvalX evaluation framework.
"""

import asyncio
from evalx import EvaluationSuite, MetricSuite, TraditionalMetrics, LLMJudge


async def example_agentic_evaluation():
    """Example of using the agentic evaluation interface."""
    print("🤖 Agentic Evaluation Example")
    print("=" * 50)
    
    # Create evaluation suite from natural language instruction
    suite = EvaluationSuite.from_instruction(
        "Evaluate my chatbot responses for helpfulness and accuracy",
        validation_level="production"
    )
    
    # Sample data
    data = [
        {
            "input": "What's the capital of France?",
            "output": "The capital of France is Paris.",
            "reference": "Paris is the capital city of France."
        },
        {
            "input": "How do I bake a chocolate cake?",
            "output": "Mix flour, sugar, eggs, and cocoa powder. Bake at 350°F for 30 minutes.",
            "reference": "To bake a chocolate cake, combine dry ingredients, add wet ingredients, and bake at 350°F for 25-30 minutes."
        }
    ]
    
    # Run evaluation
    results = await suite.evaluate_async(data)
    
    # Display results
    print(results.summary())
    print("\nDetailed Results:")
    for result in results.metric_results:
        print(f"  {result.metric_name}: {result.numeric_value:.3f}")
        if result.explanation:
            print(f"    Explanation: {result.explanation}")


def example_fine_grained_control():
    """Example of using fine-grained metric control."""
    print("\n🔧 Fine-Grained Control Example")
    print("=" * 50)
    
    # Create custom metric suite
    suite = MetricSuite()
    
    # Add traditional metrics
    suite.add_traditional_metric("bleu_score")
    suite.add_traditional_metric("semantic_similarity", threshold=0.8)
    
    # Add LLM-as-judge metrics
    suite.add_llm_judge(
        name="accuracy",
        prompt="Rate the factual accuracy of this response: {output_text}",
        model="gpt-4",
        scale="continuous"
    )
    
    # Sample data
    data = [
        {
            "input": "What is machine learning?",
            "output": "Machine learning is a subset of AI that enables computers to learn from data.",
            "reference": "Machine learning is a branch of artificial intelligence that uses algorithms to learn from data."
        }
    ]
    
    # Run evaluation synchronously
    results = suite.evaluate(data)
    
    print(f"Overall Score: {results.overall_score:.3f}")
    print("Metric Results:")
    for result in results.metric_results:
        print(f"  {result.metric_name}: {result.numeric_value:.3f}")


async def example_research_grade():
    """Example of research-grade evaluation with statistical analysis."""
    print("\n🔬 Research-Grade Evaluation Example")
    print("=" * 50)
    
    # Import research suite
    from evalx import ResearchSuite
    
    # Create research suite
    suite = ResearchSuite(
        metrics=["accuracy", "helpfulness", "bleu", "semantic_similarity"],
        validation_datasets=["squad"],
        statistical_tests=["t_test", "bootstrap"],
        human_validation=False  # Set to True if you have human annotations
    )
    
    # Generate more sample data for statistical analysis
    data = []
    for i in range(10):
        data.append({
            "input": f"Question {i+1}: What is the capital of country {i+1}?",
            "output": f"The capital is City {i+1}.",
            "reference": f"City {i+1} is the capital."
        })
    
    # Run research-grade evaluation
    results = await suite.evaluate_research_grade(data)
    
    print("Research Results:")
    print(f"Overall Score: {results.overall_score:.3f}")
    
    # Display statistical results
    for metric_name, stats in results.statistical_results.items():
        print(f"\n{metric_name} Statistics:")
        print(f"  Mean: {stats.mean:.3f} ± {stats.std:.3f}")
        print(f"  95% CI: [{stats.confidence_interval[0]:.3f}, {stats.confidence_interval[1]:.3f}]")
        if stats.effect_size:
            print(f"  Effect size: {stats.effect_size:.3f}")


def example_traditional_metrics():
    """Example of using traditional metrics directly."""
    print("\n📊 Traditional Metrics Example")
    print("=" * 50)
    
    from evalx.core.types import EvaluationInput
    
    # Create individual metrics
    bleu_metric = TraditionalMetrics.bleu_score()
    similarity_metric = TraditionalMetrics.semantic_similarity()
    
    # Sample input
    eval_input = EvaluationInput(
        output_text="The quick brown fox jumps over the lazy dog.",
        reference_text="A quick brown fox leaps over a lazy dog."
    )
    
    # Evaluate with individual metrics
    bleu_result = bleu_metric.evaluate(eval_input)
    similarity_result = similarity_metric.evaluate(eval_input)
    
    print("Individual Metric Results:")
    print(f"BLEU Score: {bleu_result.numeric_value:.3f}")
    print(f"  {bleu_result.explanation}")
    
    print(f"Semantic Similarity: {similarity_result.numeric_value:.3f}")
    print(f"  {similarity_result.explanation}")


async def example_llm_judge():
    """Example of using LLM-as-judge metrics."""
    print("\n🎭 LLM-as-Judge Example")
    print("=" * 50)
    
    # Create custom LLM judge
    custom_judge = LLMJudge.create(
        name="creativity",
        prompt="""
        Rate the creativity of this response on a scale of 0.0 to 1.0:
        
        Input: {input_text}
        Output: {output_text}
        
        Consider originality, uniqueness, and innovative thinking.
        Provide your score as a number between 0.0 and 1.0.
        """,
        model="gpt-4",
        scale="continuous",
        few_shot_examples=[
            {
                "input_text": "Write a story about a cat",
                "output_text": "The cat sat on the mat.",
                "score": 0.2,
                "reasoning": "Very basic and unoriginal story."
            },
            {
                "input_text": "Write a story about a cat", 
                "output_text": "In a world where gravity worked backwards, Luna the cat had to hold onto her food bowl to prevent it from floating away.",
                "score": 0.9,
                "reasoning": "Highly creative concept with unique world-building."
            }
        ]
    )
    
    # Sample input
    from evalx.core.types import EvaluationInput
    eval_input = EvaluationInput(
        input_text="Write a short poem about technology",
        output_text="Silicon dreams and digital streams, where data flows like morning beams. In circuits bright and screens aglow, the future's seeds begin to grow."
    )
    
    # Evaluate
    result = await custom_judge.evaluate_async(eval_input)
    
    print("LLM Judge Results:")
    print(f"Creativity Score: {result.numeric_value:.3f}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Explanation: {result.explanation}")


async def main():
    """Run all examples."""
    print("🚀 EvalX Framework Examples")
    print("=" * 70)
    
    # Run examples
    await example_agentic_evaluation()
    example_fine_grained_control()
    await example_research_grade()
    example_traditional_metrics()
    await example_llm_judge()
    
    print("\n✅ All examples completed!")
    print("\nNext steps:")
    print("1. Try modifying the prompts and metrics")
    print("2. Add your own custom metrics")
    print("3. Experiment with different models")
    print("4. Run research-grade evaluations on your data")


if __name__ == "__main__":
    # Note: You'll need to set environment variables for API keys:
    # export OPENAI_API_KEY="your_openai_key"
    # export ANTHROPIC_API_KEY="your_anthropic_key"
    
    asyncio.run(main()) 