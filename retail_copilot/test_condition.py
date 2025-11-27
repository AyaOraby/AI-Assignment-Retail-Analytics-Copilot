def test_fixed_condition():
    question = "During 'Summer Beverages 1997' as defined in the marketing calendar, which product category had the highest total quantity sold? Return {category:str, quantity:int}."
    question_lower = question.lower()
    
    print(f"Question: {question_lower}")
    print()
    
    # Test the FIXED condition
    has_category = 'category' in question_lower
    has_quantity = 'quantity' in question_lower  
    has_summer = 'summer' in question_lower
    has_highest = 'highest' in question_lower
    
    tests = [
        ("'category'", "'category' in question_lower", has_category),
        ("'quantity'", "'quantity' in question_lower", has_quantity),
        ("'summer'", "'summer' in question_lower", has_summer),
        ("'highest'", "'highest' in question_lower", has_highest),
        ("FULL FIXED CONDITION", "all four", has_category and has_quantity and has_summer and has_highest)
    ]
    
    for desc, code, result in tests:
        print(f"{desc:20} | {code:45} | {result}")
    
    print(f"\nSubstring positions:")
    print(f"'category' position: {question_lower.find('category')}")
    print(f"'quantity' position: {question_lower.find('quantity')}")
    print(f"'summer' position: {question_lower.find('summer')}")
    print(f"'highest' position: {question_lower.find('highest')}")

if __name__ == "__main__":
    test_fixed_condition()