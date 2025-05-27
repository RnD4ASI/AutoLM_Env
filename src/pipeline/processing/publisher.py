import pandas as pd
import markdown2
import re
from atlassian import Confluence
from typing import List, Tuple, Optional
from src.pipeline.shared.logging import get_logger

logger = get_logger(__name__)

class ConfluencePublisher:
    """
    A class to handle publishing content to Confluence with support for Markdown and LaTeX formatting.
    """
    def __init__(
        self,
        confluence_url: str,
        username: str,
        api_token: str
    ):
        """
        Initialize the Confluence publisher.

        Args:
            confluence_url: The base URL of your Confluence instance
            username: Confluence username (usually email)
            api_token: Confluence API token
        """
        self.confluence = Confluence(
            url=confluence_url,
            username=username,
            password=api_token,
            cloud=True  # Assuming Confluence Cloud
        )
        logger.info("Confluence publisher initialized")

    @staticmethod
    def convert_latex_to_confluence_math(text: str) -> str:
        """
        Convert inline LaTeX math (delimited by $...$) to Confluence math macro format.

        Args:
            text: Text containing LaTeX math expressions

        Returns:
            Text with LaTeX converted to Confluence math macros
        """
        pattern = r'\$(.+?)\$'
        replacement = (r'<ac:structured-macro ac:name="math">'
                    r'<ac:plain-text-body><![CDATA[\1]]></ac:plain-text-body>'
                    r'</ac:structured-macro>')
        return re.sub(pattern, replacement, text)

    @staticmethod
    def markdown_to_html(text: str) -> str:
        """
        Convert Markdown text to HTML and process inline LaTeX math.

        Args:
            text: Markdown formatted text

        Returns:
            HTML formatted text with processed LaTeX
        """
        html = markdown2.markdown(text, extras=["fenced-code-blocks"])
        html = ConfluencePublisher.convert_latex_to_confluence_math(html)
        return html

    def generate_section_html(
        self,
        section_title: str,
        dataframe: pd.DataFrame,
        content_column: str = "Content",
        first_section: bool = False
    ) -> str:
        """
        Generate HTML for a section.

        Args:
            section_title: Title of the section
            dataframe: DataFrame containing content
            content_column: Name of the column containing markdown
            first_section: If True, includes table of contents

        Returns:
            HTML formatted section content
        """
        df = dataframe.copy()
        df[content_column] = df[content_column].apply(self.markdown_to_html)
        
        if first_section:
            toc_macro = (
                '<ac:structured-macro ac:name="toc" ac:schema-version="1">'
                '<ac:parameter ac:name="printable">false</ac:parameter>'
                '</ac:structured-macro>'
            )
            df_toc = pd.DataFrame({content_column: [toc_macro]}, index=["Table of Content"])
            df_with_toc = pd.concat([df, df_toc])
            table_html = df_with_toc.to_html(columns=[content_column], index=True, header=False, escape=False)
        else:
            table_html = df.to_html(columns=[content_column], escape=False, index=False)
        
        section_html = f"<h2>{section_title}</h2><hr>{table_html}"
        return section_html

    def build_full_page_html(
        self,
        page_title: str,
        sections: List[Tuple[str, pd.DataFrame, bool]]
    ) -> str:
        """
        Build the full HTML content for the Confluence page.

        Args:
            page_title: The main title of the page
            sections: List of tuples (section_title, dataframe, first_section_flag)

        Returns:
            Complete HTML content for the page
        """
        full_html = f"<h1>{page_title}</h1>"
        for section_title, df, is_first in sections:
            section_html = self.generate_section_html(section_title, df, first_section=is_first)
            full_html += section_html
        return full_html

    def publish_page(
        self,
        page_title: str,
        content: str,
        space_key: str,
        parent_id: Optional[str] = None
    ) -> dict:
        """
        Publish content to a Confluence page.

        Args:
            page_title: Title of the page
            content: HTML content to publish
            space_key: Confluence space key
            parent_id: Optional parent page ID

        Returns:
            Response from Confluence API
        """
        try:
            # Check if page exists
            existing_page = self.confluence.get_page_by_title(
                space=space_key,
                title=page_title
            )

            if existing_page:
                # Update existing page
                page_id = existing_page['id']
                response = self.confluence.update_page(
                    page_id=page_id,
                    title=page_title,
                    body=content,
                    parent_id=parent_id,
                    type='page',
                    representation='storage'
                )
                logger.info(f"Updated Confluence page: {page_title}")
            else:
                # Create new page
                response = self.confluence.create_page(
                    space=space_key,
                    title=page_title,
                    body=content,
                    parent_id=parent_id,
                    type='page',
                    representation='storage'
                )
                logger.info(f"Created new Confluence page: {page_title}")

            return response

        except Exception as e:
            logger.error(f"Failed to publish to Confluence: {str(e)}")
            raise

# ---------------------------
# Example usage:
# ---------------------------
if __name__ == "__main__":
    # Define a sample markdown string with a subsubtitle, bullet points, and inline math.
    sample_markdown = """
    #### Subsubtitle Example

    - Bullet point one
    - Bullet point two

    Here is an inline math formula: $E=mc^2$
    """
    
    # Create sample DataFrames for two sections.
    # For the first section, we want to show row headers and add a TOC row.
    df_section1 = pd.DataFrame({
        "Content": [sample_markdown, sample_markdown]
    })
    # For the second section, use the standard table.
    df_section2 = pd.DataFrame({
        "Content": [sample_markdown, sample_markdown]
    })
    
    # Define sections as a list of tuples:
    # (section title, dataframe, first_section_flag)
    sections = [
        ("Section 1: Overview", df_section1, True),   # First section with special table
        ("Section 2: Detailed Data", df_section2, False)
    ]
    
    # Initialize the Confluence publisher
    publisher = ConfluencePublisher(
        confluence_url="https://your-domain.atlassian.net/wiki",
        username="your-email@example.com",
        api_token="your-api-token"
    )
    
    # Build the complete page HTML.
    page_title = "Complex Formatted Confluence Page with TOC in Table"
    full_page_html = publisher.build_full_page_html(page_title, sections)
    
    # For testing purposes, save the generated HTML to a file.
    with open("generated_confluence_page.html", "w", encoding="utf-8") as f:
        f.write(full_page_html)
    print("Generated HTML saved to generated_confluence_page.html")
    
    # To publish to Confluence, uncomment and provide your credentials:
    # space_key = "YOUR_SPACE_KEY"
    # response = publisher.publish_page(
    #     page_title,
    #     full_page_html,
    #     space_key,
    #     parent_id=None  # or specify a parent page ID if desired
    # )
    # if response.get("id"):
    #     print("Page published successfully with ID:", response["id"])
    # else:
    #     print("Error creating page:", response)