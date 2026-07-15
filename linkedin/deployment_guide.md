# Deployment Guide: Get a URL for LinkedIn

You have two good deployment paths.

---

## Option A: Streamlit Community Cloud URL

Use this when you want the project to be interactive and editable through app controls.

1. Create a GitHub account if you do not already have one.
2. Create a new repository named:

```text
virtual-etf-portfolio-optimizer
```

3. Upload all files from this project folder.
4. Go to Streamlit Community Cloud.
5. Sign in and connect your GitHub account.
6. Choose your repository.
7. Set the main file path to:

```text
app.py
```

8. Deploy the app.
9. Copy the generated Streamlit URL.
10. Add that URL to your LinkedIn post and LinkedIn Featured section.

---

## Option B: GitHub Pages Static Report URL

Use this when you want a clean web report that loads instantly and does not require the app to run.

1. Push the repository to GitHub.
2. Open the repository.
3. Go to **Settings**.
4. Go to **Pages**.
5. Select **Deploy from a branch**.
6. Choose:

```text
Branch: main
Folder: /docs
```

7. Save.
8. Your public report URL will look like:

```text
https://YOUR_USERNAME.github.io/virtual-etf-portfolio-optimizer/
```

---

## Best Recommendation

Use both:

- Streamlit URL for the interactive app
- GitHub Pages URL for the polished static project report

Then put the GitHub Pages URL in your LinkedIn post and the Streamlit URL in the comments or Featured section.
