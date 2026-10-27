"""Site model: URIs and response sizes in the style of the EClog shop."""

from __future__ import annotations

import numpy as np

HOST = "https://shop.our-internet-company.pl"
STATIC_EXT = ["jpg", "png", "css", "js"]


class Site:
    def __init__(self, cfg: dict, beacon_uri: str, rng: np.random.Generator):
        self.n_products = cfg["n_products"]
        self.n_categories = cfg["n_categories"]
        self.beacon_uri = beacon_uri
        self.rng = rng
        # popularity: a few products get most of the views
        self.product_p = rng.dirichlet(np.full(self.n_products, 0.3))
        self.category_p = rng.dirichlet(np.full(self.n_categories, 0.5))

    def uri(self, action: str, seo_search: bool = False) -> str:
        r = self.rng
        if action == "home":
            return "/"
        if action == "product":
            return f"/p-{r.choice(self.n_products, p=self.product_p) + 1}.html"
        if action == "category":
            c = r.choice(self.n_categories, p=self.category_p) + 1
            page = r.integers(2, 6) if r.random() < 0.3 else 0
            return f"/c-{c}.html" + (f"/s={page}" if page else "")
        if action == "search":
            q = r.integers(1, 3000)
            return f"/wyszukiwanie-query{q}.html" if seo_search else f"/szukaj.html/szukaj=query{q}/opis=tak"
        if action == "cart":
            return r.choice(["/koszyk.html", "/inne/do_koszyka.php", "/inne/do_koszyka_ilosc.php"], p=[0.5, 0.3, 0.2])
        if action == "checkout":
            return r.choice(["/zamowienie-logowanie.html", "/zamowienie-podsumowanie.html",
                             "/zamowienie-potwierdzenie.html", "/inne/zmiana_platnosci.php"])
        if action == "account":
            return r.choice(["/logowanie.html", "/inne/logowanie.php", "/inne/rejestracja.php"])
        if action == "info":
            return r.choice([f"/polec-znajomemu-produkt.html/produkt={r.integers(1, self.n_products)}",
                             f"/kartapdf-{r.integers(1, self.n_products)}.html",
                             "/informacje-o-dostawie.html", "/our-internet-shop-about-us.html",
                             f"/napisz-recenzje-rw-{r.integers(1, self.n_products)}.html"])
        if action == "error":
            return r.choice(["/brak-produktu.html", "/brak-strony.html", "/brak-kategorii.html"])
        if action == "beacon":
            return self.beacon_uri
        if action == "static":
            ext = r.choice(STATIC_EXT, p=[0.6, 0.25, 0.1, 0.05])
            return f"/zdjecia/{r.integers(1, 20000)}.{ext}" if ext in ("jpg", "png") else f"/szablony/shop/{ext}/main.{ext}"
        raise ValueError(action)

    def response_bytes(self, action: str, medians: dict) -> int:
        med = medians[action]
        return 0 if med == 0 else int(self.rng.lognormal(np.log(med), 0.4))
