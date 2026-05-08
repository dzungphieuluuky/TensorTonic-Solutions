def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration (Synchronous Update).
    """
    # Initialize a list to hold the updates for this step
    new_values = [0.0] * len(values)
    
    for state_idx in range(len(values)):
        best_value = -float("inf")
        
        # Calculate Q(s, a) for each action
        for action_idx in range(len(transitions[state_idx])):
            expected_future_value = 0
            
            for next_state_idx in range(len(transitions[state_idx][action_idx])):
                prob = transitions[state_idx][action_idx][next_state_idx]
                
                # USE 'values' HERE (the old values), NOT 'new_values'
                expected_future_value += prob * values[next_state_idx]
            
            # Bellman Equation: R(s, a) + gamma * sum(P * V_old)
            action_value = rewards[state_idx][action_idx] + gamma * expected_future_value
            best_value = max(best_value, action_value)
            
        new_values[state_idx] = best_value
        
    return new_values