

class tests:

    def __init__(self):
        self.a = 1

    def PredGuid(self):
        iterating = True
        while iterating:
            self.a = self.a + 1
            if self.a >= 10:
                iterating = False

        return (self.a)

    def values(self):
        final_value = self.PredGuid()
        print(final_value)

        return (final_value)

tests_object = tests()
final = tests_object.values()
print(final)