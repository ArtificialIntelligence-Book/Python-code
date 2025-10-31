def triangular_mf(x, a, b, c):
    """
    Triangular membership function.
    :param x: Input value
    :param a, b, c: Triangle points (a < b < c)
    :return: Membership degree in [0,1]
    """
    if x <= a or x >= c:
        return 0
    elif x == b:
        return 1
    elif x < b:
        return (x - a) / (b - a)
    else: # x > b
        return (c - x) / (c - b)

class FuzzyVariable:
    def __init__(self, name, terms):
        """
        :param name: Variable name
        :param terms: dict of term_name: (a,b,c) triangular MF params
        """
        self.name = name
        self.terms = terms  # e.g. {'low': (0,0,5), 'medium': (0,5,10), 'high': (5,10,10)}
    
    def fuzzify(self, x):
        """
        Compute membership degree for all terms for input x
        """
        memberships = {}
        for term, (a, b, c) in self.terms.items():
            memberships[term] = triangular_mf(x, a, b, c)
        return memberships

def fuzzy_inference(temp_val, humidity_val):
    """
    Simple fuzzy inference example: input temperature and humidity
    and infer fan speed: low, medium, high
    
    Rules:
     - If temperature is high or humidity is high, fan speed is high.
     - If temperature is medium and humidity is medium, fan speed is medium.
     - Else fan speed is low.
    """

    temperature = FuzzyVariable('Temperature', {
        'low': (0, 0, 15),
        'medium': (10, 20, 30),
        'high': (25, 40, 40)
    })

    humidity = FuzzyVariable('Humidity', {
        'low': (0, 0, 40),
        'medium': (30, 50, 70),
        'high': (60, 100, 100)
    })

    fan_speed = FuzzyVariable('FanSpeed', {
        'low': (0, 0, 30),
        'medium': (20, 50, 80),
        'high': (70, 100, 100)
    })

    temp_memberships = temperature.fuzzify(temp_val)
    hum_memberships = humidity.fuzzify(humidity_val)

    # Rule evaluation using min/max (Mamdani inference)

    high_rule = max(temp_memberships['high'], hum_memberships['high'])
    medium_rule = min(temp_memberships['medium'], hum_memberships['medium'])
    low_rule = 1 - max(high_rule, medium_rule)  # Default fallback

    # Defuzzification using weighted average of centroids (simplified)
    def centroid(a, b, c):
        return (a + b + c) / 3
    
    low_c = centroid(*fan_speed.terms['low'])
    med_c = centroid(*fan_speed.terms['medium'])
    high_c = centroid(*fan_speed.terms['high'])

    numerator = low_rule * low_c + medium_rule * med_c + high_rule * high_c
    denominator = low_rule + medium_rule + high_rule

    output = numerator / denominator if denominator != 0 else 0

    return output

# Example usage:
temp_input = 28
humidity_input = 65
fan_speed_output = fuzzy_inference(temp_input, humidity_input)
print(f"Fuzzy Logic output fan speed for temp={temp_input}, humidity={humidity_input}: {fan_speed_output:.2f}")