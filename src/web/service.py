class TitleService:
    def __init__(self, predictor):
        self.predictor = predictor

    def predict(self, description: str) -> str:
        # Implement your prediction logic here
        # For demonstration purposes, we'll just return the description as the category
        category = self.predictor.predict(description)
        return category['category']